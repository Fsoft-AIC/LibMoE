from .register import register_moe
from .moe import MoeLayer

import torch
import torch.nn as nn
import torch.nn.functional as F
from moe_model.model.multimodal_encoder.siglip_smoe import SiglipMLP
import copy
@register_moe("smoesgx")
class smoesgx(MoeLayer):
    def __init__(self, in_embed_dim=768, out_embed_dim=768, num_of_experts=4, num_selected=2, expert=None, args=None):
        super().__init__(in_embed_dim, out_embed_dim, num_of_experts, num_selected, expert, args)
        
        self.log_metrics = {}
        self.is_vision = False
        self.std_gate = getattr(self.args, "std_gate", 0.02)
        self.num_selected, self.num_of_experts = self.num_selected - 1, self.num_of_experts - 1
        self.gate = nn.Linear(self.in_embed_dim, self.num_of_experts, bias=False)
        
        self.init_gate_weights(self.std_gate)
        self.init_expert_weights()
        
    def init_share_gate_global(self, gate):
        self.gate_global = gate
    def topk_expert(self, gate_logits, x):
        """
        Selects the top-k experts based on the gating logits.

        This method computes the softmax of the gating logits to obtain the probabilities,
        then selects the top-k experts with the highest probabilities for each input sample.

        Args:
            gate_logits (torch.Tensor): The logits from the gating network.

        Returns:
            tuple:
                - weights (torch.Tensor): The softmax probabilities of the top-k experts.
                - selected_experts (torch.Tensor): Indices of the top-k experts.
                - gate_softmax (torch.Tensor): The softmax probabilities for all experts.
        """
        
        # if self.is_vision : 
        #     gate_softmax = gate_logits * F.softmax(self.gate_global(x), dim=-1, dtype=torch.float32)
        # else:
        gate_softmax = F.softmax(gate_logits, dim=-1, dtype=torch.float32)
        weights, selected_experts = torch.topk(gate_softmax, self.num_selected )

        # breakpoint()
        # weights = weights[:, :, : self.num_selected]
        # selected_experts = selected_experts[:, :, -self.num_selected:]
        
        return weights, selected_experts, gate_softmax
    def forward(self, x,  return_id_experts = False, is_vision=False):
        self.is_vision = is_vision
        gate_logits = self.gate(x) 
        # if is_vision : 
        #     gate_logits += self.gate_global(x)
        # gate_logits/= 2
        weights, selected_experts, gate_softmax = self.topk_expert(gate_logits=gate_logits, x = x)
        # weights, selected_experts, gate_softmax, gate_logits = self.competition_policy(x=x)
        weights = weights / torch.sum(weights, dim=-1, keepdim=True).to(x.dtype)
        output = torch.zeros(x.shape[0], x.shape[1], self.out_embed_dim, device=x.device, dtype=x.dtype)
        # output = self.compute_moe_dynamic(gate_logits = gate_softmax, selected_experts = selected_experts, weights = weights, results = output, x = x)
        output = self.compute_moe(selected_experts, weights, output, x)
        auxiliary_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        balance_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        infor_aux = {}
        # compute loss
        if x.requires_grad or return_id_experts: 
            auxiliary_loss, balance_loss, router_z_loss = self.combine_loss(selected_experts, gate_softmax, gate_logits)
            infor_aux = {
                "balance_loss": balance_loss.clone().detach(),
                "router_z_loss": router_z_loss.clone().detach()
            }
        if self.gate.weight.requires_grad == False and return_id_experts == True:
            auxiliary_loss, balance_loss, router_z_loss = self.combine_loss(selected_experts, gate_softmax, gate_logits)
            self.log_metrics['weights'] = weights
            self.log_metrics['balance_loss'] = balance_loss.item()
            self.log_metrics['router_z_loss'] = router_z_loss.item()
            self.log_metrics['gate_softmax'] = gate_softmax
            self.log_metrics['selected_experts'] = selected_experts
        if self.is_vision:
            output = F.normalize(self.experts[-1](output), p=2.0, dim=-1, eps=1e-6)
            # output = self.experts[-1](output)

        return output, auxiliary_loss, None, infor_aux
