from typing import Optional

import torch
from torch import nn


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_paged_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    cache = kwargs.pop("cache", None)
    if cache is not None:
        key, value = cache.update(key, value, module.layer_idx, **kwargs)

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


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