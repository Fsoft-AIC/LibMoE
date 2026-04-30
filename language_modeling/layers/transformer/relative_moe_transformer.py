
from typing import Optional, List, Union, Tuple
import torch
import torch.nn
import torch.nn.functional as F
from .transformer import ActivationFunction
from .multi_head_attention import AttentionMask
from layers import MoE
import math
from framework import utils
from framework.layers import LoggingLayer
from .full_moe_relative_attention import FullMoeRopeAttention
from .fast_rope_attention import FastRopeAttention
from ..cvmm import CVMMSel
from .transformer import reset_prenorm_params


class RelativeMoeTransformerEncoderLayer(LoggingLayer, torch.nn.Module):
    def __init__(self, d_model, nhead, n_experts: int, expert_size: int, n_layers: int, layer_idx: int, dim_feedforward=2048,
                 dropout=0.1, activation: ActivationFunction = F.relu, attention_dropout=0,
                  pkm_heads: int=1,
                 selection_mode: str = "add",
                 perplexity_reg: float = 0.0,
                 n_heads: int = 1,perplexity_reg_mode: str="step",
                 head_projection_size: Optional[int] = None,
                 activation_after_topk: bool = False,
                 mlp_selection: bool = False,
                 sel_bias: bool = False,
                 bias: bool = False,  preln: bool = True,
                 moe_dropout_factor: float = 1.0,
                 drop_expert: float = 0.0,  sync_distributed: bool = True,
                 moe_init_scale: float = 1.0,
                 mla_attention: bool = False,       # multihead latent attention
                 moe_attention: bool = False,
                 moe_att_n_experts: int = 4, moe_att_expert_dropout: Optional[float] = None,
                 moe_att_selection_mode: str = "sigmoid",
                 moe_att_k: Optional[int] = None, moe_att_ppl_reg: Optional[float] = None,
                 q_expert: bool = True, k_expert: bool = True, v_expert: bool = True,
                 o_expert: bool = True,
                 v_projection_size: Optional[int] = None,
                 moe_att_separate_kq_sel: bool = False,
                 rotate_fraction: float = 0.5, rope_base: float = 10000,
                 moe_att_norm_init: bool = False,
                 moe_att_same_sel: bool = False, moe_selection_dropout: float = 0.0,
                 moe_att_selection_dropout: float = 0.0,
                 att_perplexity_reg_mode: Optional[str] = None,
                 log_interval: Optional[int] = 100,
                 nonorm: bool = False,
                 layer_std_constant: float = 2.0):
        super().__init__()
        self.preln = preln
        self.i = 0

        std_scale = math.sqrt(layer_std_constant / n_layers) if preln else 1.0
        std_scale *= math.sqrt(moe_init_scale)

        if moe_attention:
            self.self_attn = FullMoeRopeAttention(
                d_model, nhead, dropout=attention_dropout,
                projection_size=head_projection_size, init_std_scale=math.sqrt(2 / n_layers) if preln else 1.0,
                n_experts=moe_att_n_experts,
                perplexity_reg=perplexity_reg if moe_att_ppl_reg is None else moe_att_ppl_reg,
                expert_dropout=drop_expert if moe_att_expert_dropout is None else moe_att_expert_dropout,
                selection_mode=moe_att_selection_mode, q_expert=q_expert, k_expert=k_expert, v_expert=v_expert,
                moe_k=pkm_heads if moe_att_k is None else moe_att_k, o_expert=o_expert,
                v_projection_size=v_projection_size,
                separate_kq_sel=moe_att_separate_kq_sel,
                rotate_fraction=rotate_fraction, rope_base=rope_base,
                normalize_init=moe_att_norm_init,
                same_sel=moe_att_same_sel, selection_dropout=moe_att_selection_dropout,
                perplexity_reg_mode=att_perplexity_reg_mode if att_perplexity_reg_mode is not None else perplexity_reg_mode,
                log_interval=log_interval
            )
        elif mla_attention:
            mla_config = MLAConfig(
                d_model, nhead, dropout=attention_dropout, projection_size=head_projection_size)
            self.self_attn = MultiheadLatentAttention(mla_config, layer_idx)
        else:
            # self.self_attn = FastRelativeAttention(d_model, nhead, dropout=attention_dropout, projection_size=head_projection_size,)
            self.self_attn = FastRopeAttention(
                    d_model, nhead, dropout=attention_dropout,
                    projection_size=head_projection_size, rotate_fraction=rotate_fraction,
                    rope_base=rope_base)

        self.pkm = MoE(
            d_model, n_experts, expert_size, dropout=dropout * moe_dropout_factor,
            weight_scale=std_scale, selection_mode=selection_mode,
            perplexity_reg=perplexity_reg,  n_heads=n_heads,
             perplexity_reg_mode=perplexity_reg_mode,
            activation_after_topk=activation_after_topk,
             activation=activation,
             sel_bias=sel_bias, bias=bias,
             expert_dropout=drop_expert,
             sync_distributed=sync_distributed,
            selection_dropout=moe_selection_dropout,
            log_interval=log_interval,
            )

        if nonorm:
            self.self_attn.norm = torch.nn.LayerNorm(d_model)

            old_compute_sel = self.self_attn.compute_sel
            old_project = self.self_attn.project

            def compute_sel_norm(self, curr_state: torch.Tensor, attend_to: torch.Tensor):
                same_src_dest = curr_state is attend_to
                curr_state = self.norm(curr_state)
                attend_to = curr_state if same_src_dest else self.norm(attend_to)
                return old_compute_sel(curr_state, attend_to)

            def project(self, name: str, src: torch.Tensor, sel):
                if name in {"k", "q"}:
                    src = self.norm(src)
                return old_project(name, src, sel)


            self.self_attn.compute_sel = compute_sel_norm.__get__(self.self_attn)
            self.self_attn.project = project.__get__(self.self_attn)

            if mlp_selection:
                self.pkm.sel = torch.nn.Sequential(
                    torch.nn.LayerNorm(d_model),
                    self.pkm.sel
                )
            else:
                old_sel = self.pkm.sel
                self.pkm.norm = torch.nn.LayerNorm(d_model)
                self.pkm.sel = lambda x: old_sel(self.pkm.norm(x))

            self.norm1 = lambda x: x
            self.norm2 = lambda x: x
        else:
            self.norm1 = torch.nn.LayerNorm(d_model)
            self.norm2 = torch.nn.LayerNorm(d_model)

        self.dropout = torch.nn.Dropout(dropout)

        self.activation = activation

        if preln:
            reset_prenorm_params(self, n_layers)


    def forward(self, src: torch.Tensor, mask: Optional[AttentionMask] = None, attend_to: Optional[torch.Tensor] = None,
                pos_offset: Optional[int] = None) -> torch.Tensor:

        ninput = self.norm1(src) if self.preln else src
        src2 = self.self_attn(ninput, self.norm1(attend_to) if attend_to is not None else ninput, mask,
                              pos_offset=pos_offset)
        src = src + self.dropout(src2)

        mlp_input = src

        if self.preln:
            src2 = self.norm2(mlp_input)
        else:
            src2 = self.norm1(mlp_input)
            src = src2

        src3 = self.pkm(src2)

        src = src + self.dropout(src3)

        if not self.preln:
            src = self.norm2(src)

        if self.training:
            self.i += 1

        return src
