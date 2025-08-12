from typing import Callable,Optional,Union
import torch
from torch import nn
from torch.nn import functional as F
from attn_implementation import eager_paged_attention_forward
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import MoeCausalLMOutputWithPast,MoeModelOutputWithPast
from gpt_config import GptOssConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.utils.generic import OutputRecorder

class RMSNorm(nn.Module):
    def __init__(self,
                dim,
                eps=1e-8):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self,x):
        rms = torch.sqrt(torch.mean(x**2,dim=1,keepdim=True)+self.eps)
        return self.weight*(x/rms)

class GptOssRMSNorm(nn.Module):
    def __init__(self,
                dim,
                eps=1e-8):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self,x):
        variance = x.pow(2).mean(-1,keepdim=True)
        x = x * torch.sqrt(variance + self.eps)
        x = self.weight * x
        return x

class GPTOssExperts(nn.Module):
    def __init__(self,
                config):
        super().__init__()

        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.expert_dim = config.intermediate_size

        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts,self.hidden_size,2 * self.expert_dim))
        self.gate_up_proj_bias = nn.Parameter(torch.empty(self.num_experts,2 * self.expert_dim))

        self.down_proj = nn.Parameter(torch.empty((self.num_experts,self.expert_dim,self.hidden_size)))
        self.down_proj_bias = nn.Parameter(torch.empty(self.num_experts,self.hidden_size))

        self.alpha = 1.702
        self.limit = 7.0

    def forward(self,
                hidden_state,
                router_indices,
                routing_weights):
        
        batch_size = hidden_state.shape[0]
        hidden_state = hidden_state.view(-1,self.hidden_size)
        num_experts = routing_weights.shape[1]

        if self.training:
            next_states = torch.zeros_like(hidden_state,dtype=hidden_state.dtype,device=hidden_state.device)
            with torch.no_grad():
                expert_mask = torch.nn.functional.one_hot(router_indices,num_classes=num_experts)
                expert_mask = expert_mask.permute(2,1,0)

                expert_hit = torch.greater(expert_mask.sum(dim=(-1,-2)),0).nonzero()
            
            for expert_idx in expert_hit[:]:
                with torch.no_grad():
                    _, token_idx = torch.where(expert_mask[expert_idx[0]])
                current_state = hidden_state[token_idx]

                gate_up = current_state @ self.gate_up_proj[expert_idx] + self.gate_up_proj_bias[expert_idx]
                gate,up = gate_up[...,::2],gate_up[...,1:2]
                gate = gate.clamp(min=None,max=self.limit)

                up = up.clamp(min=-self.limit,max=self.limit)
                glu = gate * torch.sigmoid(gate * self.alpha)

                gated_output = (up + 1) * glu

                out = gated_output @ self.down_proj[expert_idx] + self.down_proj_bias[expert_idx]
                weighted_output = out[0] * routing_weights[token_idx,expert_idx,None]
                next_states.index_add_(0,token_idx,weighted_output.to(hidden_state.dtype))
            next_states = next_states.view(batch_size,-1,self.hidden_size)

        
        else:
            hidden_state = hidden_state.repeat(num_experts,1)
            hidden_state = hidden_state.view(num_experts,-1,self.hidden_size)

            gate_up = torch.bmm(hidden_state,self.gate_up_proj) + self.gate_up_proj_bias[...,None,:]
            gate,up = gate_up[...,::2],gate_up[...,1::2]
            gate = gate.clamp(min=None,max=self.limit)
            up = up.clamp(min=-self.limit,max=self.limit)

            glu = gate * torch.sigmoid(gate * self.alpha)

            next_states = torch.bmm(((up + 1)*glu),self.down_proj)

            next_states = next_states + self.down_proj_bias[...,None,:]
            next_states = next_states.view(num_experts,batch_size,-1,self.hidden_size)
            next_states = next_states * routing_weights.transpose(0,1).view(num_experts,batch_size,-1)[...,None]
            next_states = next_states.sum(dim=0)
        return next_states

class GPTOssTopKRouter(nn.Module):
    def __init__(self,config):
        super().__init__()

        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts

        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty(self.num_experts,self.hidden_dim))
        self.bias = nn.Parameter(torch.empty(self.num_experts))

    def forward(self,hidden_state):
        hidden_state = hidden_state.reshape(-1,self.hidden_dim)
        router_logits = F.linear(hidden_state,self.weight,self.bias)

        router_top_value,router_indices = torch.topk(router_logits,self.top_k,dim=-1)
        router_top_value = torch.nn.functional.softmax(router_top_value,dim=-1,dtype=router_top_value.dtype)

        router_scores = torch.zeros_like(router_logits).scatter_(1,router_indices,router_top_value)

        return router_scores,router_indices

class GptOssMLP(nn.Module):
    def __init__(self,
                config):
        super().__init__()
        
        self.router = GPTOssTopKRouter(config)
        self.experts = GPTOssExperts(config)

    def forward(self,hidden_states):
        router_scores,router_indices = self.router(hidden_states)
        routed_out = self.experts(hidden_states,router_indices=router_indices,routing_weights=router_scores)
        return routed_out,router_scores

class GptOssRotartEmbedding(nn.Module):
    def __init__(self,
                config,
                device=None):
        super().__init__()

        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"

        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq,self.attenstion_scaling = self.rope_init_fn(self.config,device)

        self.register_buffer("inv_freq",inv_freq,presistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self,x,position_ids):
        inv_freq_expanded = self.inv_freq[None,:,None].float().expant(position_ids.shape[0],-1,1).to(x.device)
        postion_ids_expanded = position_ids[:,None,:].float()

        device_type = x.device.type if isinstance(x.device.type,str) and x.device.type != "mps" else "cpu"

        with torch.autocast(device_type=device_type,enabled=False):
            freqs = (inv_freq_expanded.float() @ postion_ids_expanded.float()).transpose(1,2)
            emb = freqs
            cos = emb.cos() * self.attenstion_scaling
            sin = emb.sin() * self.attenstion_scaling

        return cos.to(x.dtype),sin.to(x.dtype)


def repeat_kv(hidden_state,n_rep):
    batch,num_key_value_heads,slen,head_dim = hidden_state.shape
    if n_rep == 1:
        return hidden_state
    hidden_state = hidden_state[:,:,None,:,:].expand(batch,num_key_value_heads,n_rep,slen,head_dim)
    return hidden_state.reshape(batch,num_key_value_heads*n_rep,slen,head_dim)


def _apply_rotary_emb(x,
                    cos,
                    sin):
    first_half,second_half = torch.chunk(x,2,dim=-1)
    first_ = first_half * cos - second_half * sin
    second_ = second_half * cos + first_half * sin
    return torch.cat((first_,second_),dim=-1)


def apply_rotart_pos_emb(q,k,cos,sin,position_ids=None,unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = _apply_rotary_emb(q,cos,sin)
    k_embed = _apply_rotary_emb(k,cos,sin)
    return q_embed,k_embed

def eager_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask,
    scaling,
    dropout=0.0,
    **kwargs
):
    key_state = repeat_kv(key,module.num_key_value_groups)
    value_states = repeat_kv(value,module.num_key_value_groups)
    attn_weights = torch.matmul(query,key_state.transpose(2,3)) * scaling

    if attention_mask is not None:
        casual_mask = attention_mask[:,:,:,:key_state.shape[-2]]
        attn_weights = attn_weights + casual_mask
    
    sinks = module.sinks.reshape(1,-1,1,1).expand(query.shape[0],-1,query.shape[-2],-1)
    combined_logits = torch.cat([attn_weights,sinks],dim=-1)

    combined_logits = combined_logits - combined_logits.max(dim=-1,keepdim=True).value_states
    probs = F.softmax(combined_logits,dim=-1,dtype=combined_logits.dtype)

    scores = probs[...,:-1]
    attn_weights = nn.functional.dropout(scores,p=dropout,training=module.training)
    attn_output = torch.matmul(attn_weights,value_states)
    attn_output = attn_output.transpose(1,2).contiguous()
    return attn_output,attn_weights

class GptOssAttention(nn.Module):
    def __init__(
        self,
        config,
        layer_idx
    ):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx
        
        self.head_dim = getattr(config,"head_dim",config.hidden_size//config.num_attention_heads)
        print(self.config)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim*-0.5

        self.attention_dropout = config.attention_dropout
        self.is_casual = True

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias = config.attention_bias
        )

        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias = config.attention_bias
        )

        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias = config.attention_bias
        )

        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias
        )

        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None
        self.sinks = nn.Parameter(torch.empty(config.num_attention_heads))


    def forward(self,
                hidden_state,
                position_embeddings,
                attention_mask,
                past_key_values=None,
                cache_position=None,
                **kwargs):
        
        input_shape = hidden_state.shape[:-1]
        hidden_shape = (*input_shape,-1,self.head_dim)

        query_states = self.q_proj(hidden_state).view(hidden_shape).transpose(1,2)
        key_states = self.k_proj(hidden_state).view(hidden_shape).transpose(1,2)
        value_states = self.v_proj(hidden_state).view(hidden_shape).transpose(1,2)

        cos,sin = position_embeddings
        query_states , key_states = apply_rotart_pos_emb(query_states,key_states,cos,sin)

        # if past_key_values is not None:
        #     cache_kwargs = {"cache_position":cache_position}
        #     key_state,value_states = past_key_values.update(key_state,value_states,self.layer_idx,cache_kwargs)

        attention_inference: Callable = eager_attention_forward
        # if self.config._attn_implementation != "eager":
        #     # attention_inference = ALL_ATTENTION_FUNCTION[self.config._attn_implementation]
        attention_inference = eager_paged_attention_forward

        attn_output,attn_weights = attention_inference(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window = self.sliding_window,
            s_aux=self.sinks,
            **kwargs
        )
        attn_output = attn_output.reshape(*input_shape,-1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output,attn_weights


class GptOssDecoderLayer(nn.Module):
    def __init__(self,
                config,
                layer_idx):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.self_attn = GptOssAttention(config=config,layer_idx=layer_idx)
        self.mlp = GptOssMLP(config)
        self.input_layernorm = GptOssRMSNorm(config.hidden_size,eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]
        self.post_attention_layernorm = GptOssRMSNorm(config.hidden_size,eps=config.rms_norm_eps)


    def forward(self,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                use_cache,
                cache_position,
                position_embeddings,
                **kwargs):
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_state=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs
        )
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states, _ = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class GptOssPreTrainedModel(PreTrainedModel):
    config: GptOssConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GptOssDecoderLayer"]
    _skip_keys_device_placement=['past_key_values']
    _supports_flash_attn=True
    _supports_sdpa=False
    _supports_flex_attn=True

    _can_compile_fullgraph=True
    _supports_attention_backend=True
    _can_record_outputs = {
        "router_logits":OutputRecorder(GPTOssTopKRouter,index=0),
        "hidden_states":GptOssDecoderLayer,
        "attentions":GptOssAttention,
    }

    _keep_in_fp32_modules=["post_attention_layernorm","input_layernorm","norm"]
    _supports_flask_attention=False
    _supports_flex_attention=False


    def _init_weights(self,module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0,std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module,nn.Parameter):
            module.data.normal_(mean=0.0,std=std)
        elif isinstance(module,nn.Embedding):
            module.weight.data.normal_(mean=0.0,std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module,GptOssRMSNorm):
            module.weights.data.fill_(1.0)
        elif isinstance(module,GPTOssExperts):
            module.gate_up_proj.data.normal_(mean=0.0,std=std)
            module.gate_up_proj_bias.data.zero_()
            module.down_proj.data.normal_(mean=0.0,std=std)
            module.down_proj_bias.data.zero_()

        elif isinstance(module,GptOssAttention):
            module.sinks.data.normal_(mean=0.0,std=std)
        elif isinstance(module,GPTOssTopKRouter):
            module.weight.data.normal_(mean=0.0,std=std)
            module.bias.data.zero_()
    

class GptOssModel(GptOssPreTrainedModel):
    def __init__(self,
                config):
        super().__init__()

        self.padding_idx = config.padding_idx
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size,config.hidden_size,self.padding_idx)

        self.layers = nn.ModuleList(
            [GptOssDecoderLayer(config,layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GptOssRMSNorm(config.hidden_size,eps=config.rms_norm_eps)
        self.rotary_emb = GptOssRotartEmbedding(config=config)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                use_cahce=None,
                cache_position=None,
                **kwargs):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if not isinstance(casual_mask_mapping := attention_mask,dict):
            mask_kwargs = {
                "config":inputs_embeds,
                "input_embeds": inputs_embeds,
                "attention_mask":attention_mask,
                "cache_position":cache_position,
                "past_key_values":past_key_values
            }

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states,position_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cahce,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs
            )
        hidden_states = self.norm(hidden_states)
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values
        )
    
def load_balancing_loss_func(
    gate_logits: Union[torch.Tensor, tuple[torch.Tensor], None],
    num_experts: Optional[int] = None,
    top_k=2,
    attention_mask: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, int]:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://huggingface.co/papers/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits:
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts:
            Number of experts
        top_k:
            The number of experts to route per-token, can be also interpreted as the `top-k` routing
            parameter.
        attention_mask (`torch.Tensor`, *optional*):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


class GptOssForCausalLM(GptOssPreTrainedModel,GenerationMixin):
    def __init__(self,
                 config):
        super().__init__()

        self.model = GptOssModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size,config.vocab_size,bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok

    def set_decoder(self,decoder):
        self.model = decoder    
    
    def get_decoder(self):
        return self.model
    
    def forward(self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                input_embeds=None,
                labels=None,
                use_cache=None,
                output_router_logits=None,
                cache_position=None,
                logits_to_keep=None,
                **kwargs):
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            input_embeds=input_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            cache_position=cache_position,
            **kwargs
        )

        hidden_states = outputs.last_hidden_state

        slice_indices = slice(-logits_to_keep,None) if isinstance(logits_to_keep,int) else logits_to_keep
        logits = self.lm_head(hidden_states[:,slice_indices,:])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits,labels,self.vocab_size,**kwargs)
        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)
        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits
        )