import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from typing import Optional
import sys
from ..multi_head_attention import AttentionMask
from framework.layers import RegularizedLayer


class ACTWrapper(RegularizedLayer, nn.Module):
    def __init__(self, mod, d_model: int, threshold=0.999, halting_dropout=0, act_loss: float = 0.0):
        super(ACTWrapper, self).__init__()
        self._gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(halting_dropout),
            nn.Linear(d_model, 2, bias=False)
        )
        nn.init.zeros_(self._gate[-1].weight)
        self.threshold = threshold
        self.mod = mod
        self.act_loss = act_loss

    def gate(self, h):
        logits = self._gate(h)
        return F.log_softmax(logits, dim=-1)

    def update_halting(self, log_g, log_never_halt):
        log_halt = log_never_halt[..., None] + log_g
        log_never_halt = log_halt[..., 0]
        p = torch.exp(log_halt[..., 1])
        return p, log_never_halt

    def forward(self, state, src: torch.Tensor, mask: Optional[AttentionMask] = None,
                attend_to: Optional[torch.Tensor] = None, pos_offset: Optional[int] = None):

        pad_mask = mask.src_length_mask if mask is not None else None
        if pad_mask is not None:
            pad_mask = pad_mask.float()

        if state is None:
            # In the first step, src is the embedding output. In the next step, it will be the halt-gated
            # output from the previous step.

            prev_h = src
            log_never_halt = acc_expect_depth = \
                torch.zeros_like(prev_h[..., 0])
            acc_h = torch.zeros_like(prev_h)
            i = 0
            p_never_halt = pad_mask
        else:
            (i, log_never_halt, acc_h, acc_expect_depth, prev_h) = state
            log_g = self.gate(prev_h)
            p, log_never_halt = self.update_halting(log_g, log_never_halt)
            acc_h = acc_h + p[..., None] * prev_h
            acc_expect_depth = acc_expect_depth + i * p
            p_never_halt = log_never_halt.exp()
            p_never_halt = p_never_halt.masked_fill((p_never_halt < (1 - self.threshold)), 0)
            if pad_mask is not None:
                p_never_halt = p_never_halt * pad_mask
            p_never_halt = p_never_halt.contiguous()
            i = i + 1

        if attend_to is None:
            attend_to = src

        outputs = self.mod(prev_h, mask=mask, attend_to=attend_to, pos_offset=pos_offset, halt_mask=p_never_halt)
        curr_h = outputs

        curr_act_state = (i, log_never_halt, acc_h, acc_expect_depth, curr_h)

        if state is not None:
            self_attn_input = torch.where(
                p_never_halt[..., None] < (1 - self.threshold),
                attend_to[:, -prev_h.shape[1]:],
                (acc_h + p_never_halt[..., None] * curr_h).type_as(attend_to)
            )
            act_loss = (acc_expect_depth + p_never_halt * i)
            if pad_mask:
                act_loss = act_loss * pad_mask
                cnt = pad_mask.sum()
            else:
                cnt = act_loss.numel()

            act_loss = act_loss.sum() / cnt
            self.add_reg(lambda: self.act_loss * act_loss, "act_loss")
        else:
            self_attn_input = curr_h

        return curr_act_state, self_attn_input
