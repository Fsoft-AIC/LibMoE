from .register import register_moe
from .moe import MoeLayer

import torch
import torch.nn as nn
import torch.nn.functional as F
from moe_model.model.multimodal_encoder.siglip_smoe import SiglipMLP

@register_moe("smoe")
class SMoeLayer(MoeLayer):
    def __init__(self, in_embed_dim=768, out_embed_dim=768, num_of_experts=4, num_selected=2, expert=None, args=None):
        super().__init__(in_embed_dim, out_embed_dim, num_of_experts, num_selected, expert, args)
        
        self.log_metrics = {}
        self.is_vision = False
        self.std_gate = getattr(self.args, "std_gate", 0.02)
        self.init_gate_weights(self.std_gate)
        self.init_expert_weights()
        
    def init_select_layer(self, id):
        active_dict = {
            0: 6,
            1: 6,
            2: 6,
            3: 6,
            4: 5,
            5: 5,
            6: 4,
            7: 4,
            8: 3,
            9: 3,
            10: 2,
            11: 2,
            12: 2,
            13: 2,
            14: 2,
            15: 2,
            16: 2,
            17: 2,
            18: 2,
            19: 2,
            20: 2,
            21: 2,
            22: 2,
            23: 2,
            24: 2,
            25: 2,
            26: 2
        }
        try:
            if self.args.moe_name == "smoe": 
                self.E1 = 4
                self.EL = 1
                self.num_selected  = 1
                current_device = torch.cuda.current_device()
                if current_device == 0:
                    print(f"Layer {id}: active {self.num_selected}\n")
                    with open("./logs_active.txt", "a") as f:
                        f.write(f"Layer {id}: active {self.num_selected}\n")
        except:
            print("error") 
        # self.init_expert_weights()
        # self.init_gate_weights()
    def forward(self, x,  return_id_experts = False, is_vision=False):
        self.is_vision = is_vision
        gate_logits = self.gate(x)
        weights, selected_experts, gate_softmax = self.topk_expert(gate_logits=gate_logits)
        # weights, selected_experts, gate_softmax, gate_logits = self.competition_policy(x=x)
        weights = weights / torch.sum(weights, dim=-1, keepdim=True).to(x.dtype)
        output = torch.zeros(x.shape[0], x.shape[1], self.out_embed_dim, device=x.device, dtype=x.dtype)
        # output = self.compute_moe_dynamic(gate_logits = gate_softmax, selected_experts = selected_experts, weights = weights, results = output, x = x)
        output = self.compute_moe(selected_experts, weights, output, x, return_topk_outputs=False)
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
            self.log_metrics['router_magine'] =weights[:, :, 0] - weights[:, :, 1]

        return output, auxiliary_loss, None, infor_aux
