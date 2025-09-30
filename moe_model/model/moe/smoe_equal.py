from .register import register_moe
from .moe import MoeLayer

import torch
import torch.nn as nn
import torch.nn.functional as F
from moe_model.model.multimodal_encoder.siglip_smoe import SiglipMLP
from .utils import unique_each_row_vectorized, create_table_from_index_and_value

@register_moe("smoe_equa")
class smoe_equa(MoeLayer):
    def __init__(self, in_embed_dim=768, out_embed_dim=768, num_of_experts=4, num_selected=2, expert=None, args=None):
        super().__init__(in_embed_dim, out_embed_dim, num_of_experts, num_selected, expert, args)
        
        self.log_metrics = {}
        self.is_vision = False
        self.init_gate_weights()
        self.init_expert_weights()
    def competition_policy(self, x):
        """
        Implements the competition policy for expert selection.

        Args:
            x (tensor): Input tensor of shape (B, N, D), where:
                - B: Batch size
                - N: Sequence length
                - D: Input feature dimension

        Returns:
            weights (tensor): Tensor of shape (B, N, num_selected) representing the normalized weights for the selected experts.
            selected_experts (tensor): Tensor of shape (B, N, num_selected) containing the indices of the selected experts.
            affinity_softmax (tensor): Softmax probabilities of the affinity scores, with shape (B, N, num_of_experts).
        """
        B, N, D = x.shape

        # Initialize affinity scores tensor
        affinity_scores = torch.zeros(B, N, self.num_of_experts, device=x.device, dtype=x.dtype)

        # Calculate affinity scores based on the norm of each expert's output
        for i in range(self.num_of_experts):
      
            out_i = self.experts[i](x)
            affinity_scores[:, :, i] = torch.mean(F.softmax(out_i), dim = -1)
   
        # Compute softmax of the affinity scores
        affinity_softmax = F.softmax(affinity_scores, dim=-1, dtype=torch.float)
        # Select top experts based on affinity scores
        weights, selected_experts = torch.topk(affinity_scores, self.num_selected)
        weights = weights / torch.sum(weights, dim=-1, keepdim=True).to(x.dtype)
        # weights = F.softmax(weights, dim=-1).to(x.dtype)
        return weights, selected_experts, affinity_softmax, affinity_scores
    def compute_moe_infer(self, selected_experts, weights, results, x):
        """
        Compute the output by routing through the selected experts.

        Args:
            selected_experts (torch.Tensor): Indices of the selected experts.
            weights (torch.Tensor): Weights of the selected experts.
            results (torch.Tensor): Tensor to store the results.
            x (torch.Tensor): Input tensor to be processed by the experts.

        Returns:
            torch.Tensor: The computed output from the selected experts.
        """
        infor_experts = {}
        
        for i in range(len(self.experts)):
            batch_idx, token_idx, topk_idx = torch.where(selected_experts == i)
            infor_experts[i] = [batch_idx, token_idx, topk_idx]
        output_experts = [None] * len(self.experts)
        luna_gating = torch.full(
            (x.shape[0], x.shape[1], self.num_of_experts),
            # -float('inf'),
            0,
            device=x.device,
            dtype=x.dtype
        )
        origin_gating = torch.full(
            (x.shape[0], x.shape[1], self.num_of_experts),
            # -float('inf'),
            0,
            device=x.device,
            dtype=x.dtype
        )
        weights = weights.to(x.dtype)
        for i, expert in enumerate(self.experts):
            batch_idx, token_idx, topk_idx = infor_experts[i]
            out_exp = expert(x[batch_idx, token_idx])
            output_experts[i] = out_exp
            luna_gating[batch_idx, token_idx, i] =  torch.norm(F.softplus(out_exp), dim=-1)
            
            origin_gating[batch_idx, token_idx, i] = weights[batch_idx, token_idx, topk_idx]
        luna_gating_softmax = torch.softmax(luna_gating, dim=-1)     
        luna_gating_softmax = luna_gating / torch.sum(luna_gating, dim=-1, keepdim=True).to(x.dtype)
           
        for i, expert in enumerate(self.experts):
            batch_idx, token_idx, topk_idx = infor_experts[i]
            results[batch_idx, token_idx] += luna_gating_softmax[batch_idx, token_idx, i].unsqueeze(0).T * output_experts[i] 
        return results
    def compute_moe_dynamic(self, gate_logits, selected_experts, weights, results, x):
        # gate_logits = F.softmax(gate_logits, dim=-1)
        with torch.no_grad():
            x_padded = F.pad(selected_experts, pad=(0, 0, 2-1, 0), mode='constant', value=-1) 
            
            x_unfold = x_padded.unfold(dimension=1, size=2, step=1)  

            selected_experts_stride = x_unfold.squeeze(-1).squeeze(2)  # Kích thước (B, N, L)

            index_expert_unique = unique_each_row_vectorized(selected_experts_stride, -1)
            weight_after_unique = create_table_from_index_and_value(index_expert_unique, value=gate_logits, fill_value=-float('inf'))
        
        # weights = F.softmax(weight_after_unique, dim=2)
        weights = weight_after_unique / torch.sum(weight_after_unique, dim=-1, keepdim=True).to(x.dtype)
        infor_experts = {}
        for i in range(self.num_of_experts):
            batch_idx, token_idx, topk_idx = torch.where(index_expert_unique== i)
            infor_experts[i] = [batch_idx, token_idx, topk_idx]
            
        for i, expert in enumerate(self.experts):
            batch_idx, token_idx, topk_idx = infor_experts[i]
            weights_slice = weights[batch_idx, token_idx, topk_idx].unsqueeze(0).T
            results[batch_idx, token_idx] +=weights_slice*expert(x[batch_idx, token_idx])
        return results
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
        gate_softmax = F.softmax(gate_logits, dim=-1, dtype=torch.float32)
        weights, selected_experts = torch.topk(gate_softmax, self.num_selected)
        # weights = weights[:, :, :2]
        # selected_experts = selected_experts[:, :, 2:]
        # breakpoint()
        return weights, selected_experts, gate_softmax
    def forward(self, x,  return_id_experts = False, is_vision=False):
        # breakpoint()
        self.is_vision = is_vision
        gate_logits = self.gate(x)
        weights, selected_experts, gate_softmax = self.topk_expert(gate_logits=gate_logits)
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

            # self.log_metrics['weights'] = weights
            # self.log_metrics['balance_loss'] = balance_loss.item()
            # self.log_metrics['router_z_loss'] = router_z_loss.item()
            # self.log_metrics['gate_softmax'] = gate_softmax
            # self.log_metrics['selected_experts'] = selected_experts

        return output, auxiliary_loss, None, infor_aux
