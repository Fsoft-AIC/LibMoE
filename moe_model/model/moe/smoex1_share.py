from .register import register_moe
from .moe import MoeLayer

import torch
import torch.nn as nn
import torch.nn.functional as F
from moe_model.model.multimodal_encoder.siglip_smoe import SiglipMLP
import copy
@register_moe("smoesx1_share")
class smoesx1_share(MoeLayer):
    def __init__(self, in_embed_dim=768, out_embed_dim=768, num_of_experts=4, num_selected=2, expert=None, args=None):
        super().__init__(in_embed_dim, out_embed_dim, num_of_experts, num_selected, expert, args)
        
        self.log_metrics = {}
        self.is_vision = False
        self.std_gate = getattr(self.args, "std_gate", 0.02)
        self.E1 = 4
        self.EL = 1
        self.num_selected, self.num_of_experts = self.num_selected - 1, self.num_of_experts - 1
        self.gate = nn.Linear(self.in_embed_dim, self.num_of_experts, bias=False)
        
        self.init_gate_weights(self.std_gate)
        self.init_expert_weights()
        
    def init_select_layer(self, id):
        self.num_selected  = int(self.E1 - (id - 1) * (self.E1 - self.EL) / 28)
        current_device = torch.cuda.current_device()
        if current_device == 0:
            print(f"Layer {id}: active {self.num_selected}\n")
            with open("./logs_active.txt", "a") as f:
                f.write(f"Layer {id}: active {self.num_selected}\n")
        
    def forward(self, x,  return_id_experts = False, is_vision=False):
        self.is_vision = is_vision
        gate_logits = self.gate(x) 
        weights, selected_experts, gate_softmax = self.topk_expert(gate_logits=gate_logits)
        weights = weights / torch.sum(weights, dim=-1, keepdim=True).to(x.dtype)
        output = torch.zeros(x.shape[0], x.shape[1], self.out_embed_dim, device=x.device, dtype=x.dtype)
        output_selected = self.compute_moe(selected_experts, weights, output, x)
        auxiliary_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        balance_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        infor_aux = {}
        output_shared = self.experts[self.num_of_experts](x)
        
        output += output_shared*0.5 + output_selected*0.5
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
   
        return output, auxiliary_loss, None, infor_aux
