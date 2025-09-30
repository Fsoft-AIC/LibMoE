import math
from typing import Optional, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from .multi_head_attention import AttentionMask
from .MLA_rotary_embedding import DeepseekV2RotaryEmbedding, DeepseekV2LinearScalingRotaryEmbedding, \
    DeepseekV2DynamicNTKScalingRotaryEmbedding, DeepseekV2YarnRotaryEmbedding, DeepseekV2RMSNorm, \
    apply_rotary_pos_emb, yarn_get_mscale




@dataclass
class MLAConfig:
    # modify version of DeepseekV2Config for smaller model (158M -> 1330M params)
    # modify from: /home/fpt/moeut_training_code/layers/transformer/deepseekv2_config_256B.json
    # and from the config of the model in codebase
    def __init__(self, state_size: int, n_heads: int, dropout: float = 0.0, input_size: Optional[int] = None, projection_size: Optional[int] = None, output_size: Optional[int] = None):
        # state_size = 512/1024/1536

        # rope scaling config
        self.rope_scaling = {
            "type": "yarn",
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 40,
            "mscale": 0.707,
            "original_max_position_embeddings": 4096,
        }

        # attention config
        self.attention_dropout = dropout
        self.hidden_size = state_size
        self.num_attention_heads = n_heads

        self.max_position_embeddings = 163840

        self.rope_theta = 10000
        self.q_lora_rank = 64
        self.qk_rope_head_dim = 64
        self.kv_lora_rank = 64
        self.v_head_dim = 64
        self.qk_nope_head_dim = 64

        self.attention_bias = False


# Modify from DeepseekV2Attention at https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/main/modeling_deepseek.py
class MultiheadLatentAttention(torch.nn.Module):
    """ Multihead Latent Attention """

    def __init__(self, config: MLAConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.is_causal = True

        self.q_a_proj = nn.Linear(
            self.hidden_size, config.q_lora_rank, bias=config.attention_bias
        )
        self.q_a_layernorm = DeepseekV2RMSNorm(config.q_lora_rank)
        self.q_b_proj = nn.Linear(
            config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
        )

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = DeepseekV2RMSNorm(config.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            self.num_heads
            * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )
        self._init_rope()

        self.softmax_scale = self.q_head_dim ** (-0.5)
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = DeepseekV2RotaryEmbedding(
                self.qk_rope_head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = DeepseekV2LinearScalingRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = DeepseekV2DynamicNTKScalingRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "yarn":
                kwargs = {
                    key: self.config.rope_scaling[key]
                    for key in [
                        "original_max_position_embeddings",
                        "beta_fast",
                        "beta_slow",
                        "mscale",
                        "mscale_all_dim",
                    ]
                    if key in self.config.rope_scaling
                }
                self.rotary_emb = DeepseekV2YarnRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                    **kwargs,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.v_head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    #  make it equalvalent with FastRopeAttention: def forward(self, curr_state: torch.Tensor, attend_to: torch.Tensor, mask: Optional[AttentionMask], pos_offset: Optional[int] = None, need_weights: bool = False):
    # def forward(
    #     self,
    #     hidden_states: torch.Tensor,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     past_key_value: Optional[Cache] = None,
    #     use_cache: bool = False,
    #     **kwargs,
    # ) -> torch.Tensor:

    def forward(
        self,
        hidden_states: torch.Tensor,                 # corresponds to curr_state
        attend_to: Optional[torch.Tensor] = None,      # corresponds to attend_to
        mask: Optional[torch.Tensor] = None,           # corresponds to mask
        pos_offset: Optional[int] = None,              # corresponds to pos_offset
        need_weights: bool = False,                    # corresponds to need_weights
        **kwargs,
    ) -> torch.Tensor:
        # If attend_to is not provided, default to hidden_states
        if attend_to is None:
            attend_to = hidden_states

        # If pos_offset is not provided, make sure the sequence lengths match (otherwise an offset is needed)
        if pos_offset is None:
            if hidden_states.shape[1] != attend_to.shape[1]:
                raise ValueError(
                    "If attend_to has a different sequence length than hidden_states, pos_offset must be provided"
                )
            pos_offset = 0

        # Compute query states from hidden_states (query/hidden_states)
        bsz, q_len, _ = hidden_states.size()
        _, in_len, _ = attend_to.size()

        # ---- Query computation ----
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # ---- Key and Value computation ----
        compressed_kv = self.kv_a_proj_with_mqa(attend_to)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        # For keys with RoPE, reshape using the key sequence length (in_len)
        k_pe = k_pe.view(bsz, in_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        kv = kv.view(bsz, in_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        k_nope, value_states = torch.split(kv,
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )


        # ---- Apply Rotary Positional Embedding to the RoPE parts ----
        # Use the key/value sequence length (in_len) to get cos and sin
        cos, sin = self.rotary_emb(value_states, seq_len=in_len)
        # Create position indices for queries (enforcing that if lengths differ, pos_offset is provided)
        position_ids = torch.arange(q_len, device=hidden_states.device) + pos_offset

        # Expand k_pe if necessary:
        if k_pe.size(1) == 1 and self.num_heads > 1:
            k_pe = k_pe.expand(bsz, self.num_heads, in_len, self.qk_rope_head_dim)

        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids, unsqueeze_dim=1)

        # ---- Merge the no-RoPE and RoPE parts ----
        query_states = torch.empty(bsz, self.num_heads, q_len, self.q_head_dim, device=hidden_states.device, dtype=q_nope.dtype)
        query_states[..., :self.qk_nope_head_dim] = q_nope
        query_states[..., self.qk_nope_head_dim:] = q_pe

        key_states = torch.empty(bsz, self.num_heads, in_len, self.q_head_dim, device=attend_to.device, dtype=k_nope.dtype)
        key_states[..., :self.qk_nope_head_dim] = k_nope
        key_states[..., self.qk_nope_head_dim:] = k_pe

        # ---- Compute attention weights ----
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.softmax_scale

        # if mask is not None:
        #     breakpoint()
        #     if mask.position_mask.size() != (bsz, 1, q_len, in_len):
        #         raise ValueError(
        #             f"Attention mask should be of size {(bsz, 1, q_len, in_len)}, but is {mask.position_mask.size()}"
        #         )
        #     attn_weights = attn_weights + mask.position_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # ---- Compute the attention output ----
        attn_output = torch.matmul(attn_weights, value_states)
        if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(attn_output)

        if not need_weights:
            attn_weights = None

        return attn_output