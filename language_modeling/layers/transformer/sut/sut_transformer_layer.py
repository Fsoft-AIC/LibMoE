# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from framework.layers import RegularizedLayer
from ..multi_head_attention import AttentionMask

from .moa_attention import MultiheadAttention
from .parallel_linear.moe import MoE


class ModuleWrapper(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x)


class SUTTransformer(RegularizedLayer, nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *cfg.encoder.normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, d_model: int, dropout: float, activation_dropout: float, attention_dropout: float,
                attn_num_expert: int, attn_k: int, attn_expert_dim: int, head_dim: int, preln: bool,
                ff_expert_dim: int, ff_num_expert: int, ff_k: int,
                cvloss: float = 0,
                switchloss: float = 0, zloss: float = 0, miloss: float = 0, sample_topk: int = 0,
                gating_dropout: float = 0, activation_fn = F.relu, max_relative_positions: int = 64):
        super().__init__()

        self.embed_dim = d_model
        self.self_attn_layer_norm = torch.nn.LayerNorm(self.embed_dim)
        self.dropout_module = torch.nn.Dropout(dropout)

        self.activation_fn = activation_fn

        self.self_attn = MultiheadAttention(
            d_model,
            dropout=attention_dropout,
            self_attention=True,
            num_expert=attn_num_expert,
            top_k=attn_k,
            expert_dim=attn_expert_dim,
            head_dim=head_dim,
            cvloss=cvloss,
            switchloss=switchloss,
            zloss=zloss,
            miloss=miloss,
            sample_topk=sample_topk,
            gating_dropout=gating_dropout,
            max_positions=max_relative_positions
        )

        activation_dropout_p = activation_dropout
        self.activation_dropout_module = torch.nn.Dropout(
            float(activation_dropout_p)
        )
        self.normalize_before = preln
        self.moe = MoE(
            self.embed_dim,
            ff_expert_dim,
            ff_num_expert,
            ff_k,
            cvloss=cvloss,
            switchloss=switchloss,
            zloss=zloss,
            miloss=miloss,
            activation=nn.Sequential(
                ModuleWrapper(self.activation_fn),
                self.dropout_module
            ),
            noisy_gating=False,
            bias=True,
            acc_aux_loss=True,
            gating_dropout=gating_dropout,
        )
        self.final_layer_norm = torch.nn.LayerNorm(self.embed_dim)

    def residual_connection(self, x, residual):
        return residual + x

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def get_mask_tensor(self, src_len: int, mask: Optional[AttentionMask]) -> Optional[torch.Tensor]:
        if mask is None or (mask.position_mask is None and mask.src_length_mask is None):
            return None

        # mask.position_mask: [..., N_out, N_in]
        # mask.src_length_mask: [B, ...., N_in]
        # True where it has to be masked

        if mask.position_mask is not None:
            n_pad = src_len - mask.position_mask.shape[-1]
            if n_pad > 0:
                pm = F.pad(mask.position_mask, (n_pad, 0), 'constant', value=False)
            else:
                pm = mask.position_mask

        if mask.position_mask is None:
            m = mask.src_length_mask.unsqueeze(-2).unsqueeze(-2)
        elif mask.src_length_mask is None:
            m = pm
        else:
            m = mask.src_length_mask.unsqueeze(-2).unsqueeze(-2) | pm

        return m

    def forward(self, src: torch.Tensor, mask: Optional[AttentionMask] = None, attend_to: Optional[torch.Tensor] = None,
                pos_offset: Optional[int] = None, halt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

    # def forward(self,
    #             x, self_attn_input, halt_mask, layer_idx,
    #             encoder_padding_mask: Optional[Tensor],
    #             attn_mask: Optional[Tensor] = None):

        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """

        x = src.transpose(0,1)
        self_attn_input  = attend_to.transpose(0,1) if  attend_to is not None else x

        if halt_mask is not None:
            halt_mask = halt_mask.transpose(0,1).contiguous()


        mask = self.get_mask_tensor(self_attn_input.shape[0], mask)

        residual = x
        if self.normalize_before:
            x_ = self.self_attn_layer_norm(x)
            if x is self_attn_input:
                self_attn_input = x_
            else:
                self_attn_input = self.self_attn_layer_norm(self_attn_input)
            x = x_

        x, self_aux_loss = self.self_attn.forward(
            query=x, key=self_attn_input, value=self_attn_input,
            attn_mask=mask,
            skip_mask=halt_mask
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)

        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x, mlpexp_aux_loss = self.moe.forward(x, skip_mask=halt_mask)

        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        # The losses are 0.0 if all the regularizers are turned off
        if torch.is_tensor(self_aux_loss):
            self.add_reg(lambda: self_aux_loss, "attention_aux_loss")
        if torch.is_tensor(mlpexp_aux_loss):
            self.add_reg(lambda: mlpexp_aux_loss, "mlpexp_aux_loss")

        return x.transpose(0,1)
