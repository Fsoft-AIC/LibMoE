
import torch
import torch.nn as nn
from einops import rearrange, repeat, reduce, pack, unpack
import torch.nn.functional as F

from .register import register_moe
from .moe import MoeLayer


class SigmoidGating(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(SigmoidGating, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts, bias=True)
    
    def forward(self, x):
        return torch.sigmoid(-self.fc(x))


@register_moe("moe_sigmoidgating")
class MoESigmoidGating(MoeLayer):

    def __init__(self, in_embed_dim=768, out_embed_dim=768, num_of_experts=4, num_selected=2, expert=None, args=None):
        super().__init__(in_embed_dim, out_embed_dim, num_of_experts, num_selected, expert, args)
        print("sigmoid dense")
        self.gate = nn.Linear(in_embed_dim, self.num_of_experts, bias=True)
        self.sigmoid = nn.Sigmoid()
    def compute_moe(self, gate_logits, results, x):
        for i in range(len(self.experts)):
            weights_slice = gate_logits[:, :, i].reshape(gate_logits[:, :, i].shape[0], gate_logits[:, :, i].shape[1], 1)
            results += weights_slice*self.experts[i](x)

        return results
    
    def forward(self, x, return_id_experts = False):
        # compute output
        gate_logits = self.sigmoid(self.gate(x))

        output = torch.zeros(x.shape[0], x.shape[1], self.out_embed_dim, device=x.device, dtype=x.dtype)
        output = self.compute_moe(gate_logits, output, x)
        

       
        return output, torch.tensor(0.0), None