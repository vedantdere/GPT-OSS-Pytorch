from torch import nn # Very Important in Deep Learning :)
import torch
from torch.nn import functional as F
from attn_implementation import eager_paged_attention_forward
from transformers.modeling_outputs import MoeModelOutputWithPast # Handles the models output. 
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS # We are plaining to create separate pure pytorch implementation for rope 


class GptOssRMSNorm(nn.Module):
    def __init__(self,
                dim,
                eps=1e-8):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self,x):
        """
        Apply RMS Normalization to the input tensor x.
        Args:
            x (torch.Tensor): Input tensor to be normalized.
            shape of x should be (batch_size, sequence_length, hidden_size).
        Returns:
            torch.Tensor: Normalized tensor.
        """
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
        """"
        Apply the experts to the hidden state based on the routing indices and weights.
        Args:
            hidden_state (torch.Tensor): Input tensor of shape (batch_size, sequence_length, hidden_size).
            router_indices (torch.Tensor): Indices of the experts to route to, shape (batch_size, num_experts).
            routing_weights (torch.Tensor): Weights for the routing, shape (batch_size, num_experts).
        Returns:
            torch.Tensor: Output tensor after applying the experts, shape (batch_size, sequence_length, hidden_size).
        """
        
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
        """"
        Apply the top-k routing to the hidden state.
        Args:
            hidden_state (torch.Tensor): Input tensor of shape (batch_size, sequence_length, hidden_size).
        Returns:
            torch.Tensor: Router scores of shape (batch_size, sequence_length, num_experts).
            torch.Tensor: Indices of the top-k experts for each token, shape (batch_size, sequence_length, top_k).
        """
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
        """
        Apply the MLP layer with routing to the hidden states.
        Args:
            hidden_states (torch.Tensor): Input tensor of shape (batch_size, sequence_length, hidden_size).
        Returns:
            torch.Tensor: Output tensor after applying the MLP layer, shape (batch_size, sequence_length, hidden_size).
            torch.Tensor: Router scores of shape (batch_size, sequence_length, num_experts).
        """
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

        self.register_buffer("inv_freq",inv_freq)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self,x,position_ids):
        """"
        Apply Rotary Position Embeddings to the input tensor x based on position_ids.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, hidden_size).
            position_ids (torch.Tensor): Position IDs of shape (batch_size, sequence_length).
        Returns:
            torch.Tensor: Cosine and sine embeddings for the rotary position embeddings.
        """
        inv_freq_expanded = self.inv_freq[None,:,None].float().expand(position_ids.shape[0],-1,1).to(x.device)
        postion_ids_expanded = position_ids[:,None,:].float()

        device_type = x.device.type if isinstance(x.device.type,str) and x.device.type != "mps" else "cpu"

        with torch.autocast(device_type=device_type,enabled=False):
            freqs = (inv_freq_expanded.float() @ postion_ids_expanded.float()).transpose(1,2)
            emb = freqs
            cos = emb.cos() * self.attenstion_scaling
            sin = emb.sin() * self.attenstion_scaling

        return cos.to(x.dtype),sin.to(x.dtype)


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
        
        """"
        Apply the attention mechanism to the hidden state.
        Args:
            hidden_state (torch.Tensor): Input tensor of shape (batch_size, sequence_length, hidden_size).
            position_embeddings (tuple): Cosine and sine embeddings for the rotary position embeddings.
            attention_mask (torch.Tensor): Attention mask of shape (batch_size, 1, sequence_length, sequence_length).
            past_key_values (Optional): Past key values for caching.
            cache_position (Optional): Position in the cache for the current sequence.
        Returns:
            torch.Tensor: Output tensor after applying the attention mechanism, shape (batch_size, sequence_length, hidden_size).
            torch.Tensor: Attention weights of shape (batch_size, num_attention_heads, sequence_length, sequence_length).
        """
        
        input_shape = hidden_state.shape[:-1]
        hidden_shape = (*input_shape,-1,self.head_dim)

        query_states = self.q_proj(hidden_state).view(hidden_shape).transpose(1,2)
        key_states = self.k_proj(hidden_state).view(hidden_shape).transpose(1,2)
        value_states = self.v_proj(hidden_state).view(hidden_shape).transpose(1,2)

        cos,sin = position_embeddings
        query_states , key_states = apply_rotart_pos_emb(query_states,key_states,cos,sin)
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
        """
        Apply the decoder layer to the hidden states.
        Args:
            hidden_states (torch.Tensor): Input tensor of shape (batch_size, sequence_length, hidden_size).
            attention_mask (torch.Tensor): Attention mask of shape (batch_size, 1, sequence_length, sequence_length).
            position_ids (torch.Tensor): Position IDs of shape (batch_size, sequence_length).
            past_key_values (Optional): Past key values for caching.
            use_cache (bool): Whether to use caching for faster inference.
            cache_position (Optional): Position in the cache for the current sequence.
            position_embeddings (tuple): Cosine and sine embeddings for the rotary position embeddings.
        Returns:
            torch.Tensor: Output tensor after applying the decoder layer, shape (batch_size, sequence_length, hidden_size).
        """

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


class GptOssModel(nn.Module):
    def __init__(self,
                config):
        super().__init__()

        self.padding_idx = 0 #config.padding_idx
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
                use_cache=None,
                cache_position=None,
                **kwargs):
        """"
        Apply the GPT-OSS model to the input data.
        Args:
            input_ids (torch.Tensor): Input token IDs of shape (batch_size, sequence_length).
            attention_mask (torch.Tensor): Attention mask of shape (batch_size, 1, sequence_length, sequence_length).
            position_ids (torch.Tensor): Position IDs of shape (batch_size, sequence_length).
            past_key_values (Optional): Past key values for caching.
            inputs_embeds (Optional): Input embeddings of shape (batch_size, sequence_length, hidden_size).
            use_cache (bool): Whether to use caching for faster inference.
            cache_position (Optional): Position in the cache for the current sequence.
        Returns:
            MoeModelOutputWithPast: Output of the model containing the last hidden state and past key values.   
        """

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
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs
            )
        hidden_states = self.norm(hidden_states)
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values
        )
    

class GPTOssModelFull(nn.Module):
    def __init__(self,
                 config):
        super().__init__()

        self.model = GptOssModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size,config.vocab_size,bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.config = config

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
        
        """"
        Apply the GPT-OSS model to the input data and compute the logits.
        Args:
            input_ids (torch.Tensor): Input token IDs of shape (batch_size, sequence_length).
            attention_mask (torch.Tensor): Attention mask of shape (batch_size, 1, sequence_length, sequence_length).
            position_ids (torch.Tensor): Position IDs of shape (batch_size, sequence_length).
            past_key_values (Optional): Past key values for caching.
            input_embeds (Optional): Input embeddings of shape (batch_size, sequence_length, hidden_size).
            labels (Optional): Labels for computing the loss.
            use_cache (bool): Whether to use caching for faster inference.
            output_router_logits (Optional): Whether to output router logits.
            cache_position (Optional): Position in the cache for the current sequence.
            logits_to_keep (Optional): Indices of the logits to keep.
        Returns:
            torch.Tensor: Logits of shape (batch_size, sequence_length, vocab_size).
        """

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
        return logits
