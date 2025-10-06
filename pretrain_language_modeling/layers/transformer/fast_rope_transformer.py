
from typing import Optional
import torch
import torch.nn
import torch.nn.functional as F
from .multi_head_attention import AttentionMask
from .transformer import ActivationFunction
from .transformer import reset_prenorm_params
from .fast_rope_attention import FastRopeAttention


class FastRopeTransformerEncoderLayer(torch.nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation: ActivationFunction = F.relu,
                 attention_dropout=0, drop_expand: bool = True,
                 head_projection_size: Optional[int] = None, preln: bool = False, n_layers: Optional[int] = None,
                 rotate_fraction: float = 0.5, rope_base: float = 10000, parallel: bool = False):
        super().__init__()
        self.preln = preln
        self.parallel = parallel

        self.self_attn = FastRopeAttention(
            d_model, nhead, dropout=attention_dropout,
            projection_size=head_projection_size, rotate_fraction=rotate_fraction,
            rope_base=rope_base)
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout) if drop_expand else lambda x: x
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)

        if not self.parallel:
            self.norm1 = torch.nn.LayerNorm(d_model)
            self.norm2 = torch.nn.LayerNorm(d_model)
        else:
            # In case of parallel MLP and attention, we need only 1 layernorm.
            if self.preln:
                self.norm1 = torch.nn.LayerNorm(d_model)
                self.norm2 = lambda x: x
            else:
                self.norm1 = lambda x: x
                self.norm2 = torch.nn.LayerNorm(d_model)

        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.activation = activation

        if preln:
            if n_layers is None:
                raise ValueError("n_layers must be specified when using preln")
            reset_prenorm_params(self, n_layers)
        else:
            self.reset_parameters()

    def forward(self, src: torch.Tensor, mask: Optional[AttentionMask] = None, attend_to: Optional[torch.Tensor] = None,
                pos_offset: Optional[int] = None) -> torch.Tensor:
        norm_input = self.norm1(src) if self.preln else src
        src2 = self.self_attn(norm_input, self.norm1(attend_to) if attend_to is not None else norm_input, mask, pos_offset=pos_offset)
        src = src + self.dropout1(src2)

        mlp_input = norm_input if self.parallel else src
        if self.preln:
            src2 = self.norm2(mlp_input)
        else:
            src2 = self.norm1(mlp_input)
            if not self.parallel:
                src = src2

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)

        if not self.preln:
            src = self.norm2(src)
        return src

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.linear1.weight, gain=torch.nn.init.calculate_gain('relu')
                                     if self.activation is F.relu else 1.0)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
