
import torch
import torch.nn as nn
from einops import rearrange, repeat, reduce, pack, unpack
from moe_model.model.moe.register import register_moe
import torch.nn.functional as F

from .moe import MoeLayer

@register_moe("smoe_perturbed")
class MoEPerturbedCosingGating(MoeLayer):
    def __init__(self, in_embed_dim=768, out_embed_dim=768, num_of_experts=4, num_selected=2, expert=None, args=None, theta=0.1):
        super().__init__(in_embed_dim, out_embed_dim, num_of_experts, num_selected, expert, args)

        self.gate = nn.Linear(in_embed_dim, num_of_experts, bias=True)
        self.theta = theta
        
    
    def _keepTopk(self, x, beta):
        '''
        KeepTopK(v, u):

            --> v + u if v in the top K elements of vpp
            --> -inf otherwise
            
        In this code, x = v and u = beta
        '''
        x = x + beta
        weights, selected_experts  = torch.topk(x, k = self.num_selected, dim=2) 
        weights = torch.softmax(weights, dim=2)
        return weights, selected_experts 

    
    def forward(self, x, return_id_experts = False, is_vision = False):
        # compute output
        gate_logits = (x @ self.gate.weight.T) / ((torch.norm(x, dim=-1, keepdim=True) + self.theta) * (torch.norm(self.gate.weight, dim=1, keepdim=True).T + self.theta))
        
        gate_softmax = F.softmax(gate_logits, dim=-1, dtype=torch.float).to(x.dtype)
        weights, selected_experts = self._keepTopk(gate_softmax, self.gate.bias)
        output = torch.zeros(x.shape[0], x.shape[1], self.out_embed_dim, device=x.device, dtype=x.dtype)
        output = self.compute_moe(selected_experts, weights, output, x)
        
        auxiliary_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        balance_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        auxiliary_loss, balance_loss = self.combine_loss(selected_experts, gate_softmax, gate_logits)
        
        if return_id_experts:
            return output, auxiliary_loss, selected_experts, balance_loss
        else:
            return output, auxiliary_loss, None, balance_loss