class PretrainedConfig:
    model_type=""
    base_config_key=""
    sub_configs={}
    has_no_defaults_at_init=None
    attribute_map={}
    base_model_tp_plan=None
    base_model_pp_plan=None
    
    def __init__(self,
                output_hidden_state=False,
                output_attentions=False,
                return_dict=True,
                torchscript=False,
                torch_dtype=False,
                purned_heads=None,
                tie_word_embeddings=True,
                chunk_size_feed_forward=0,
                is_encoder_decoder=False,
                is_decoder=False,
                cross_attention_hidden_size=None,
                and_cross_attention=False,
                tie_encoder_decoder=False,
                tokenizer_class=None,
                prefix=None,
                bos_token_id=None,
                pad_token_id=None,
                eos_token_id=None,
                sep_token_id=None,
                decoder_start_token_id=None
                ):
        self.return_dict=return_dict
        self.output_hidden_states=output_hidden_state
        self.torchscript=torchscript
        self.torch_dtype=torch_dtype
        self._output_attention-output_attentions

        self.pruned_heads=purned_heads if purned_heads is not None else {}
        self.tie_word_embeddings=tie_word_embeddings
        self.chunk_size_feed_forward=chunk_size_feed_forward
        self.is_encoder_decoder=is_encoder_decoder
        self.is_decoder=is_decoder
        self.cross_attention_hidden_size=cross_attention_hidden_size
        self.tie_encoder_decoder=tie_encoder_decoder

        


class Config(PretrainedConfig):
    def __init__(self,
                 num_hidden_layers=36,
                 num_local_experts=128,
                 vocab_size=201088,
                 hidden_size=2880,
                 intermediate_size=2880,
                 head_dim=64,
                 num_attention_heads=64,
                 num_key_value_heads=8,
                 sliding_window=128,
                 rope_theta=150000.0,
                 tie_word_embeddings=False,
                 hidden_act='silu',
                 initializer_range=0.02,
                 max_position_embeddings=131072,
                 rms_norm_eps=1e-5,
                 rope_scaling={
                     "rope_type":"yarn",
                     "factor":32.0,
                     "beta_fast":32.0,
                     "beta_slow":1.0,
                     "truncate":False
                 },
                 attention_drop=0.0,
                 num_experts_per_tok=4,
                 router_aux_loss_coef=0.9,
                 output_router_logits=False,
                 use_cache=True,
                 layer_types=None,
                 **kwargs
                 ):

        self.vocab_size=vocab_size
        self.num_attention_heads=num_attention_heads
        self.tie_word_embeddings=tie_word_embeddings
        self.sliding_window=sliding_window
        self.hidden_act=hidden_act
        self.num_experts_per_tok=num_experts_per_tok
        self.router_aux_loss_coef=router_aux_loss_coef
        self.num_key_value_heads=num_key_value_heads
        self.hidden_size=hidden_size
        self.num_local_experts=num_local_experts
        self.initializer_range=initializer_range
        self.rms_norm_eps=rms_norm_eps
        self.intermediate_size=intermediate_size
        self.rope_theta=rope_theta
        self.num_hidden_layers=num_hidden_layers
        self.rope_scaling=rope_scaling
        self.attention_dropout=attention_drop
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        self.layer_types = layer_types

        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if bool((i+1)%2) else "full_attention" for i in range(self.num_hidden_layers)
            ]
        
        self.attention_bias=True
        self.max_position_embeddings=max_position_embeddings
        self.router_aux_loss_coef=output_router_logits
        self.output_router_logits=output_router_logits
        self.use_cache=use_cache


class GPTOss_120B(PretrainedConfig):
    def __init__(self):
        self.attention_bias=True
        self.attention_dropout=0.0
        self.eos_token_id=200002
        self.experts_per_tok=4
        self.head_dim=64
        self.intermediate_size=2880
        self.hidden_act='silu'
        self.hidden_size=2880
        self.initial_context_length=4096
        self.initializer_range=0.02
        self.intermediate_size=2880
        self.num_hidden_layers=36

        self.layer_types = [
            "sliding_attention" if bool((i+1)%2) else "full_attention" for i in range(self.num_hidden_layers)
        ]
        self.max_position_embeddings=131072
        self.model_type="gpt_oss_120b"
        self.num_attention_heads=64
        
        self.num_experts_per_tok=4
        self.num_key_value_heads=8
        self.num_local_experts=128
        self.output_router_logits=False
        self.pad_token_id=199999
        self.rms_norm_eps=1e-5
        self.rope_scaling={
            "rope_type":"yarn",
            "factor":32.0,
            "beta_fast":32.0,
            "beta_slow":1.0,
            "truncate":False
        }

        self.rope_theta=150000.0
        self.router_aux_loss_coef=0.9
        self.sliding_window=128
        self.swiglu_limit=7.0
        self.tie_word_embeddings=False
        self.use_cache=False
        self.vocab_size=201088


class GPTOss_20B(PretrainedConfig):
    def __init__(self):
        self.attention_bias=True
        self.attention_dropout=0.0
        self.eos_token_id=200002
        self.experts_per_tok=4
        self.intermediate_size=2880
        self.head_dim=64
        self.hidden_act='silu'
        self.hidden_size=2880
        self.initial_context_length=4096
        self.initializer_range=0.02
        self.intermediate_size=2880
        self.num_hidden_layers=24
        self.layer_types = [
            "sliding_attention" if bool((i+1)%2) else "full_attention" for i in range(self.num_hidden_layers)
        ]
        self.max_position_embeddings=131072
        self.model_type="gpt_oss_20b"
        self.num_attention_heads=64
        self.num_experts_per_tok=4
        self.num_key_value_heads=8
        self.num_local_experts=32
        self.output_router_logits=False
        self.pad_token_id=199999
        self.rms_norm_eps=1e-5
        self.rope_scaling={
            "rope_type":"yarn",
            "factor":32.0,
            "beta_fast":32.0,
            "beta_slow":1.0,
            "truncate":False
        }

        self.rope_theta=150000.0
        self.router_aux_loss_coef=0.9
        self.sliding_window=128
        self.swiglu_limit=7.0
        self.tie_word_embeddings=False
        self.use_cache=False
        self.vocab_size=201088

class SmallConfig(PretrainedConfig):
    def __init__(self,
                 num_hidden_layers=36,
                 num_local_experts=32,
                 vocab_size=10000,
                 hidden_size=64,
                 intermediate_size=64,
                 head_dim=4,
                 num_attention_heads=4,
                 num_key_value_heads=4,
                 sliding_window=32,
                 rope_theta=150000.0,
                 tie_word_embeddings=False,
                 hidden_act='silu',
                 initializer_range=0.02,
                 max_position_embeddings=131072,
                 rms_norm_eps=1e-5,
                 rope_scaling={
                     "rope_type":"yarn",
                     "factor":32.0,
                     "beta_fast":32.0,
                     "beta_slow":1.0,
                     "truncate":False
                 },
                 attention_drop=0.0,
                 num_experts_per_tok=2,
                 router_aux_loss_coef=0.9,
                 output_router_logits=False,
                 use_cache=True,
                 layer_types=None,
                 **kwargs
                 ):

        self.vocab_size=vocab_size
        self.num_attention_heads=num_attention_heads
        self.tie_word_embeddings=tie_word_embeddings
        self.sliding_window=sliding_window
        self.hidden_act=hidden_act
        self.num_experts_per_tok=num_experts_per_tok
        self.router_aux_loss_coef=router_aux_loss_coef
        self.num_key_value_heads=num_key_value_heads
        self.hidden_size=hidden_size
        self.num_local_experts=num_local_experts
        self.initializer_range=initializer_range
        self.rms_norm_eps=rms_norm_eps
        self.intermediate_size=intermediate_size
        self.rope_theta=rope_theta
        self.num_hidden_layers=num_hidden_layers
        self.rope_scaling=rope_scaling
        self.attention_dropout=attention_drop
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        self.layer_types = layer_types

        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if bool((i+1)%2) else "full_attention" for i in range(self.num_hidden_layers)
            ]
        
        self.attention_bias=True
        self.max_position_embeddings=max_position_embeddings
        self.router_aux_loss_coef=output_router_logits
        self.output_router_logits=output_router_logits
        self.use_cache=use_cache

