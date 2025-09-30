from moe_model.model.moe.register import register_moe
from moe_model.model.moe.moe import MoeLayer
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
import pandas as pd
import copy 
@register_moe("moeks")
class MoEKS(MoeLayer):
    def __init__(self, in_embed_dim=768, out_embed_dim=768, num_of_experts=4, num_selected=2, expert=None, args=None):
        super().__init__(in_embed_dim, out_embed_dim, num_of_experts , num_selected , expert, args)

    
        self.warm_up = args.warm_up  # Warm up expert with SMoE
        self.rate_flip = args.rate_flip
        self.total_steps = None
        self.current_steps = 0
        self.step_warm = None
        self.is_prob_flips = True
        self.hybrid = False
        self.expert_shared = copy.deepcopy(self.experts[-1])
        self.gate = nn.Linear(in_embed_dim, num_of_experts - 1, bias=False)       
        self.log_metrics = {}
        self.num_selected = num_selected - 1
        self.num_of_experts = num_of_experts - 1
        self.lamda_shared = 0.5
        self.nb_wakers = 5
        self.id_layer = 0
        self.percent_stop = 0.1
        
        self.init_gate_weights()
        self.init_expert_weights()
        # if self.args.sparse_upcycling == False:
        self.init_shared()
        # self.init_duplicate()
    def init_duplicate(self):
        for i in range(self.experts):
            self.experts[i] = copy.deepcopy(self.expert_shared)
            
        print("init expert duplicate sucessfull")
    def init_shared(self):
        if self.args.sparse_upcycling:
            if len(self.experts) == self.num_of_experts: return
            self.expert_shared = copy.deepcopy(self.experts[-1])
            self.experts = self.experts[:-1]
            print("init expert shared sucessfull")
        else:
            if len(self.experts) == self.num_of_experts: return
            self.expert_shared = self.experts[-1]
            self.experts = self.experts[:-1]
            print("init expert shared from scratch sucessfull")
        
    def set_total_steps(self, total_steps):
        self.total_steps = total_steps
        self.stop_share = total_steps - total_steps * self.percent_stop

    def set_current_steps(self, step):
        self.current_steps = step
    
    def shared_knowlegde(self, x, gate_softmax, output_shared):
        
        share_loss = None
        
        _, selected_experts = torch.topk((-1) * gate_softmax, self.num_of_experts - self.num_selected) # 3, 1, - 1 shared exp 
        for i in range(self.num_of_experts):
            batch_idx, token_idx, topk_idx = torch.where(selected_experts == i)
            if batch_idx.numel() == 0 : continue
            out_exp = self.experts[i](x[batch_idx, token_idx]).to(torch.float32)
             
            if share_loss == None:
                share_loss = F.mse_loss(out_exp, output_shared[batch_idx, token_idx])
            else:
                share_loss += F.mse_loss(out_exp, output_shared[batch_idx, token_idx])
        share_loss = share_loss / self.num_of_experts
 
        return share_loss
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
       
        return weights, selected_experts, gate_softmax
    def compute_moe(self, selected_experts, weights, results, x, expert_outputs = None, return_topk_outputs = False):
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
            idx_expanded = selected_experts.unsqueeze(-1).expand(B, N, self.num_selected, x.size(-1))
            topk_expert_outputs = torch.gather(expert_outputs_topk, dim=2, index=idx_expanded)
            diver_loss = self.experts_diversity_loss(topk_expert_outputs)
            if x.requires_grad == False: 
                self.log_metrics['diver_loss'] = diver_loss.item()
            else:
                return results, diver_loss
        return results
    def router_policy(self, x):
        """
        Implements the standard routing policy using gate logits.

        Args:
            x (tensor): Input tensor of shape (B, N, D).

        Returns:
            weights (tensor): Normalized weights of the selected experts.
            selected_experts (tensor): Indices of the selected experts.
            gate_softmax (tensor): Softmax probabilities of the gate logits.
        """

        gate_logits = self.gate(x)
        
        # Select experts using top-k gating
        weights, selected_experts, gate_softmax = self.topk_expert(gate_logits=gate_logits)
    
        weights = weights / torch.sum(weights, dim=-1, keepdim=True).to(x.dtype)

        return weights, selected_experts, gate_softmax, gate_logits
    
    def forward(self, x, return_id_experts=False, is_vision=False):
        """
        Forward pass of the CompleteSMoE model.

        Args:
            x (tensor): Input tensor of shape (B, N, D).
            return_id_experts (bool): Whether to return the selected expert indices.

        Returns:
            output (tensor): Output tensor of shape (B, N, out_embed_dim).
            auxiliary_loss (tensor): Auxiliary loss incurred during the selection process.
            selected_experts (tensor or None): Indices of the selected experts if `return_id_experts` is True, otherwise None.
        """
        self.is_vision = is_vision
        output = torch.zeros(x.shape[0], x.shape[1], self.out_embed_dim, device=x.device, dtype=x.dtype)

        # Normal gating for expert selection
        gate_weights, gate_selected_experts, gate_softmax, gate_logits = self.router_policy(x)
        
        auxiliary_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        
        balance_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        
        infor_aux = {}
        
        output_shared = self.expert_shared(x)
        
        shared_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        
        auxiliary_loss, balance_loss, router_z_loss = self.combine_loss(
            selected_experts=gate_selected_experts,
            gate_softmax=gate_softmax,
            gate_logits=gate_logits
        )
        # Perform MoE computation using gate-selected experts
        output = self.compute_moe(
            weights=gate_weights,
            selected_experts=gate_selected_experts,
            results=output,
            x=x
        )

        # Decide whether to use the competition policy based on `rate_flip`
        if x.requires_grad and (self.current_steps < self.stop_share) and (self.current_steps + 1) % (self.nb_wakers + self.id_layer) == 0:
            
            # Use competition policy for expert selection
            shared_loss = self.shared_knowlegde(
                    x = x,
                    output_shared = output_shared.to(torch.float32).detach(),
                    gate_softmax=gate_softmax                
                )
            auxiliary_loss += shared_loss*0.001
    
        
        infor_aux = {
            "balance_loss": balance_loss.clone().detach(),
            "router_z_loss": router_z_loss.clone().detach(),
            "shared_loss": shared_loss
        }
        
        output_final = output_shared * self.lamda_shared + output * (1 - self.lamda_shared )
        return output_final, auxiliary_loss, None, infor_aux 
