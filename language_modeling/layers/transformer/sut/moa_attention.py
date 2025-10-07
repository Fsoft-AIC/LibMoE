# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter

utils = None

from .parallel_linear.moe import MoE


class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=False,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        num_expert=12,
        top_k=4,
        expert_dim=256,
        head_dim=64,
        cvloss=0,
        switchloss=0,
        zloss=0,
        miloss=0,
        sample_topk=0,
        gating_dropout=0,
        max_positions=64,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.top_k = top_k
        self.dropout_module = torch.nn.Dropout(
            dropout
        )

        assert expert_dim % head_dim == 0
        self.num_heads = expert_dim // head_dim
        self.head_dim = head_dim
        self.expert_dim = expert_dim
        self.scaling = self.head_dim**-0.25
        self.max_positions = max_positions

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        self.k_proj = nn.Linear(self.kdim, self.expert_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, self.expert_dim, bias=bias)
        self.q_proj = MoE(embed_dim, self.expert_dim, num_expert, k=top_k,
                cvloss=cvloss, switchloss=switchloss, zloss=zloss, miloss=miloss,
                gating_dropout=gating_dropout,
                noisy_gating=False, acc_aux_loss=True, bias=bias)
        if self.self_attention:
            # self.rel_pos_emb = nn.Linear(self.head_dim, self.max_positions * 2 + 1, bias=False)
            self.rel_pos_emb = Parameter(torch.Tensor(self.num_heads, self.head_dim, self.max_positions * 2 + 1))

        self.sample_topk = sample_topk

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False
        self.skip_embed_dim_check = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            # nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            # nn.init.xavier_uniform_(self.q_proj.weight)

        # nn.init.xavier_uniform_(self.out_proj.weight)
        # if self.out_proj.bias is not None:
        #     nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

        if self.self_attention:
            nn.init.zeros_(self.rel_pos_emb)

    def rel_pos_logits(self, query, length, last=False):
        device = query.device
        dest_len = query.shape[-2]
        assert dest_len <= length

        idx = torch.arange(length, dtype=torch.long, device=device)
        if not last:
            # Support non-square attention
            idx_grid = idx[None, :] - idx[-dest_len:, None]
        else:
            idx_grid = idx[None, :] - (length - 1)
        idx_grid = torch.clamp(idx_grid, min=1 - self.max_positions, max=self.max_positions - 1) + self.max_positions

        # logits = self.rel_pos_emb(query)
        logits = torch.einsum('bkhid,hdj->bkhij', query, self.rel_pos_emb)

        output = logits.gather(-1, idx_grid[None, None, None, :, :].expand(logits.size(0), logits.size(1), logits.size(2), -1, -1))
        return output

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        skip_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        if not self.skip_embed_dim_check:
            assert (
                embed_dim == self.embed_dim
            ), f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]

        if self.self_attention:
            q, aux_loss = self.q_proj.map(
                query,
                skip_mask=skip_mask, sample_topk=self.sample_topk
            )
            k = self.k_proj(key)
            v = self.v_proj(value)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q, aux_loss = self.q_proj.map(
                query,
                skip_mask=skip_mask, sample_topk=self.sample_topk
                )
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)

        else:
            assert key is not None and value is not None
            q, aux_loss = self.q_proj.map(
                query,
                skip_mask=skip_mask_, sample_topk=self.sample_topk
                )
            k = self.k_proj(key)
            v = self.v_proj(value)

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )

        q = q * self.scaling
        k = k * self.scaling

        # (bsz, top_k, num_heads, seq_len, head_dim)
        q = q.reshape(tgt_len, bsz, self.top_k, self.num_heads, self.head_dim).permute(1, 2, 3, 0, 4)
        if k is not None:
            # (bsz, num_heads, seq_len, head_dim)
            k = k.reshape(-1, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        if v is not None:
            # (bsz, num_heads, seq_len, head_dim)
            v = v.reshape(-1, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

        assert k is not None
        assert k.size(2) == src_len

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = F.pad(k, (0, 0, 0, 1))
            v = F.pad(v, (0, 0, 0, 1))
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))

        total_bsz = bsz * self.top_k * self.num_heads
        attn_weights = torch.einsum('bkhie,bhje->bkhij', q, k).reshape(
            total_bsz, tgt_len, src_len)
        if self.self_attention:
            attn_weights = attn_weights + self.rel_pos_logits(
                q,
                src_len,
            ).view(total_bsz, tgt_len, src_len)
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [total_bsz, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)

            attn_weights.masked_fill_(attn_mask, float('-inf'))

        if before_softmax:
            return attn_weights, v

        attn_weights_float = F.softmax(
            attn_weights, dim=-1
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn = torch.einsum(
            'bkhij,bhje->bkhie',
            attn_probs.view(bsz, self.top_k, self.num_heads, tgt_len, src_len),
            v
        )
        assert list(attn.size()) == [bsz, self.top_k, self.num_heads, tgt_len, self.head_dim]
        attn = self.q_proj.reduce(attn.permute(3, 0, 1, 2, 4).reshape(tgt_len, bsz, self.top_k, self.expert_dim))

        return attn, aux_loss

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights
