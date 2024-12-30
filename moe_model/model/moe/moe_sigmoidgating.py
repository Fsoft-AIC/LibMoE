from .register import register_moe
from .moe import MoeLayer

import torch
import torch.nn as nn
import torch.nn.functional as F
#Warning: Changed from dense training (paper) to sparse expert activation (this version v1.1).

@register_moe("smoe_sigmoidgating")
class SMoESigmoidGating(MoeLayer):
    def __init__(self, in_embed_dim=768, out_embed_dim=768, num_of_experts=4, num_selected=2, expert=None, args=None):
        super().__init__(in_embed_dim, out_embed_dim, num_of_experts, num_selected, expert, args)
        self.gate = nn.Linear(in_embed_dim, self.num_of_experts, bias=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x,  return_id_experts = False,  is_vision = False):
        
        gate_logits = self.sigmoid(self.gate(x))
        weights, selected_experts, gate_softmax = self.topk_expert(gate_logits=gate_logits)
        weights = weights / torch.sum(weights, dim=-1, keepdim=True).to(x.dtype)
        output = torch.zeros(x.shape[0], x.shape[1], self.out_embed_dim, device=x.device, dtype=x.dtype)
        output = self.compute_moe(selected_experts, weights, output, x)
        # compute loss
        auxiliary_loss , balance_loss= self.combine_loss(selected_experts, gate_softmax, gate_logits)
        if return_id_experts:
            return output, auxiliary_loss, selected_experts, balance_loss
        else:
            return output, auxiliary_loss, None, balance_loss
