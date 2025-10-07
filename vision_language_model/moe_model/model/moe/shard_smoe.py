import torch
import torch.nn as nn
from einops import rearrange, repeat, reduce, pack, unpack
from moe_model.model.moe.register import register_moe
import torch.nn.functional as F
import copy
import loguru

from .moe import MoeLayer


@register_moe("smoe_share")
class MoEShareLayer(MoeLayer):
    def __init__(self, in_embed_dim=768, out_embed_dim=768, num_of_experts=4, num_selected=2, expert=None, args=None):
        super().__init__(in_embed_dim, out_embed_dim, num_of_experts, num_selected, expert, args)
        self.in_embed_dim = in_embed_dim
        self.out_embed_dim = out_embed_dim
        self.num_of_experts = num_of_experts
        self.num_selected = num_selected 
        self.args = args
        
        # # initialize the router and expert
        # if expert is None:
        #     print("initialize the selected expert with random init")
        #     self.experts = nn.ModuleList([
        #         nn.Sequential(nn.Linear(self.in_embed_dim, self.out_embed_dim), 
        #         nn.GELU(), 
        #         nn.Linear(self.out_embed_dim, self.out_embed_dim)) for _ in range(self.num_of_experts)])
        # else:
        #     print("Initialize the selected expert with deep copy expert")
        #     self.experts = nn.ModuleList([copy.deepcopy(expert) for _ in range(self.num_of_experts)])

        self.num_selected, self.num_of_experts = self.num_selected - 1, self.num_of_experts - 1
        self.gate = nn.Linear(self.in_embed_dim, self.num_of_experts, bias=False)
        
        self.init_gate_weights()
        self.init_expert_weights()
        
    def compute_moe(self, selected_experts, weights, results, x, expert_outputs = None, return_topk_outputs = False, output_shared = None):
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
        B, N, D = x.shape

        for i in range(len(self.experts)):
            batch_idx, token_idx, topk_idx = torch.where(selected_experts == i)
            infor_experts[i] = [batch_idx, token_idx, topk_idx]
        if return_topk_outputs == True:
            expert_outputs_topk = torch.zeros(x.shape[0], x.shape[1], self.num_of_experts, self.out_embed_dim, device=x.device, dtype=x.dtype)
        
        # if expert_outputs is not None:
        is_expert = expert_outputs is not None
        for i in range(self.num_of_experts):

            expert = self.experts[i]
       
            batch_idx, token_idx, topk_idx = infor_experts[i]
            if batch_idx.numel() == 0 : continue
            if is_expert:
                out_exp = expert_outputs[i][batch_idx, token_idx]
            else:
                out_exp = expert(x[batch_idx, token_idx])
            if return_topk_outputs == True:
                expert_outputs_topk[batch_idx, token_idx, i] = out_exp
            results[batch_idx, token_idx] += weights[batch_idx, token_idx, topk_idx].unsqueeze(0).T * out_exp
        if return_topk_outputs:
            idx_expanded = selected_experts.unsqueeze(-1).expand(B, N, selected_experts.shape[-1], results.size(-1))
            topk_expert_outputs = torch.gather(expert_outputs_topk, dim=2, index=idx_expanded)
            
            shared_expanded = output_shared.unsqueeze(2)  
            topk_expert_outputs = torch.cat([topk_expert_outputs, shared_expanded], dim=2)
            shared_expanded = shared_expanded.to(dtype=topk_expert_outputs.dtype, device=topk_expert_outputs.device)
        
            
            diver_loss = self.experts_diversity_loss(topk_expert_outputs)
            if x.requires_grad == False: 
                self.log_metrics['diver_loss'] = diver_loss.item()
            else:
                return results, diver_loss
        return results
    
    
    def forward(self, x, return_id_experts = False, is_vision = False):

        gate_logits = self.gate(x)

        output = torch.zeros(x.shape[0], x.shape[1], self.out_embed_dim, device=x.device, dtype=x.dtype)

        weights, selected_experts, gate_softmax = self.topk_expert(gate_logits=gate_logits)
        
        weights = weights / torch.sum(weights, dim=-1, keepdim=True).to(x.dtype)
        
        output_selected = torch.zeros(x.shape[0], x.shape[1], self.out_embed_dim, device=x.device, dtype=x.dtype)
        
        output_shared = self.experts[self.num_of_experts](x)
        
        output_selected = self.compute_moe(selected_experts, weights, output_selected, x, return_topk_outputs=False, output_shared = output_shared)
        
        
        
        output+= output_shared*0.5 + output_selected*0.5
        auxiliary_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        balance_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        infor_aux = {}
        if x.requires_grad: 
            # compute loss
            auxiliary_loss, balance_loss, router_z_loss = self.combine_loss(selected_experts, gate_softmax, gate_logits)
            infor_aux = {
                "balance_loss": balance_loss.clone().detach(),
                "router_z_loss": router_z_loss.clone().detach()
            }
        if self.gate.weight.requires_grad==False and return_id_experts == True:
            auxiliary_loss, balance_loss, router_z_loss = self.combine_loss(selected_experts, gate_softmax, gate_logits)
            self.log_metrics['weights'] = weights
            self.log_metrics['balance_loss'] = balance_loss.item()
            self.log_metrics['router_z_loss'] = router_z_loss.item()
            self.log_metrics['gate_softmax'] = gate_softmax
            self.log_metrics['selected_experts'] = selected_experts
            self.log_metrics['router_magine'] = weights[:, :, 0] - weights[:, :, 1]

        
        return output, auxiliary_loss, None, infor_aux
