from .register import register_moe
from .moe import MoeLayer

import torch
import torch.nn as nn
import torch.nn.functional as F
from moe_model.model.multimodal_encoder.siglip_smoe import SiglipMLP

@register_moe("remoe")
class ReMoE(MoeLayer):
    def __init__(self, in_embed_dim=768, out_embed_dim=768, num_of_experts=4, num_selected=2, expert=None, args=None):
        super().__init__(in_embed_dim, out_embed_dim, num_of_experts, num_selected, expert, args)
        
        self.log_metrics = {}
        self.is_vision = False
        self.std_gate = getattr(self.args, "std_gate", 0.02)
        self.moe_relu_sparsity = 0.0
        self.target_sparse = 1 - (self.num_selected / self.num_of_experts)
        self.moe_relu_l1_reg_coeff_multiplier = getattr(self.args, 'moe_relu_l1_reg_coeff_multiplier', 1.2)
        self.init_gate_weights(self.std_gate)
        self.init_expert_weights()
        
    def topk_expert(self, gate_logits):
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
        gate_relu = F.relu(gate_logits.float())
        # apply RELU activation 
        weights, selected_experts = torch.topk(gate_relu, self.num_selected)
    
        return weights, selected_experts, gate_relu
    def update_balance_coef(self, ):
        '''
        CONTROLLING SPARSITY VIA ADAPTIVE L1 REGULARIZATION interage to balance loss
        
        '''
        if self.moe_relu_sparsity < self.target_sparse:
            
            self.args.balance_loss_coef *= self.moe_relu_l1_reg_coeff_multiplier
            
        else:
            
            self.args.balance_loss_coef /= self.moe_relu_l1_reg_coeff_multiplier
    def update_amount_sparsity(self, probs):
        """Record level sparsity of router

        Args:
            probs: gate logit B X T X E
        """
        routing_map = probs > 0
        # Record the sparsity of the ReLU output
        sparsity = 1 - routing_map.sum().float() / routing_map.numel()
        
        self.moe_relu_sparsity = sparsity
    def forward(self, x,  return_id_experts = False, is_vision=False):
        self.is_vision = is_vision
        
        gate_logits = self.gate(x)
        
        weights, selected_experts, gate_relu = self.topk_expert(gate_logits=gate_logits)
        # should add scale avoid devided zero -> nana 
        weights = weights / (torch.sum(weights, dim=-1, keepdim=True).to(x.dtype) + 1e-6)
        
        output = torch.zeros(x.shape[0], x.shape[1], self.out_embed_dim, device=x.device, dtype=x.dtype)

        output = self.compute_moe(selected_experts, weights, output, x)
        # if torch.isnan(weights).any(): breakpoint()
        
        auxiliary_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        
        balance_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        
        infor_aux = {}
        
        # compute loss
        if x.requires_grad or return_id_experts: 
            
            # update balance coef 
            # self.update_balance_coef()
            
            # # convert logit into probability token distribution to experts (follow align with switch transformer)
            # gate_relu_prob = F.softmax(gate_relu, dim=-1, dtype=torch.float)
            gate_softmax = F.softmax(gate_logits, dim=-1, dtype=torch.float)
            # compute aux loss
            auxiliary_loss, balance_loss, router_z_loss = self.combine_loss(selected_experts, gate_softmax, gate_logits)
       
            # update level sparsity 
            # self.update_amount_sparsity(gate_relu)
            
            infor_aux = {
                # "balance_remoe_loss": balance_loss.clone().detach(),
                "balance_loss": balance_loss.clone().detach(),
                "router_z_loss": router_z_loss.clone().detach(),
                # "moe_relu_sparsity": self.moe_relu_sparsity.clone().detach(),
                # "balance_loss_coef": torch.tensor(self.args.balance_loss_coef, device=x.device, dtype=x.dtype)
            }
        
            # self.log_metrics['weights'] = weights
            # self.log_metrics['balance_loss'] = balance_loss.item()
            # self.log_metrics['router_z_loss'] = router_z_loss.item()
            # self.log_metrics['gate_softmax'] = gate_softmax
            # self.log_metrics['selected_experts'] = selected_experts
        # udapte
        return output, auxiliary_loss, None, infor_aux
