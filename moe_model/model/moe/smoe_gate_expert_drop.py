from moe_model.model.moe.register import register_moe
from moe_model.model.moe.moe import MoeLayer
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist

@register_moe("smoe_gate_drop")
class smoe_gate_expert_drop(MoeLayer):
    def __init__(self, in_embed_dim=768, out_embed_dim=768, num_of_experts=4, num_selected=2, expert=None, args=None):
        super().__init__(in_embed_dim, out_embed_dim, num_of_experts, num_selected, expert, args)

        if args is None or not hasattr(args, 'rate_flip'):
            raise ValueError("The 'args' parameter must have the attribute 'rate_flip'.")
        if not hasattr(args, 'warm_up'):
            raise ValueError("The 'args' parameter must include 'warm_up'.")
        self.warm_up = args.warm_up  # Warm up expert with SMoE

        self.rate_flip = args.rate_flip
        self.total_steps = None
        self.current_steps = 0
        self.step_warm = None
        self.is_prob_flips = True
        self.p = 0.05
        self.identity  = nn.Identity()
        # if self.training:
        self.init_gate_weights(std=self.args.std_gate)

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

    def router_loss(self, gate_softmax, affinity_softmax):
        
        """
        Computes the router loss, which encourages the gate's softmax probabilities to match the affinity scores.

        Args:
            gate_softmax (tensor): Softmax probabilities from the gate logits of shape (B, N, num_of_experts).
            affinity_softmax (tensor): Softmax probabilities of the affinity scores of shape (B, N, num_of_experts).

        Returns:
            loss (tensor): Scalar tensor representing the mean squared error (MSE) between the gate and affinity softmax probabilities.
        """
        loss = F.mse_loss(gate_softmax, affinity_softmax)
        return loss
    
    def _forward_skip(self, x):
        auxiliary_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        infor_aux = {}
        rank  = dist.get_rank()    
        out = self.experts[rank](x)
        return out, auxiliary_loss, None, infor_aux 
    def forward(self, x, return_id_experts=False, is_vision=False):
        
        
        
        """
        Forward pass of the CompleteSMoE model.

        Args:
            x (tensor): Input tensor of shape (B, N, D).
            return_id_experts (bool): Whether to return the selected expert indices.

        Returns:C
            output (tensor): Output tensor of shape (B, N, out_embed_dim).
            auxiliary_loss (tensor): Auxiliary loss incurred during the selection process.
            selected_experts (tensor or None): Indices of the selected experts if `return_id_experts` is True, otherwise None.
        """
        flag = torch.zeros(1, dtype=torch.bool, device=x.device)
        
        if dist.get_rank() == 0:
            flag.bernoulli_(self.p)          # True  -> dropout ON
            
        dist.broadcast(flag, 0)
        flag = flag.item()
        
        if flag:
            return self._forward_skip(x)
        else:
            return self._forward_normal(x, return_id_experts=return_id_experts)
    def _forward_normal(self, x, return_id_experts=False):
        
        auxiliary_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        balance_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        output = torch.zeros(x.shape[0], x.shape[1], self.out_embed_dim, device=x.device, dtype=x.dtype)

        # Normal gating for expert selection
        gate_weights, gate_selected_experts, gate_softmax, gate_logits = self.router_policy(x)

        
        
        # Perform MoE computation using gate-selected experts
        output = self.compute_moe(
            weights=gate_weights,
            selected_experts=gate_selected_experts,
            results=output,
            x=x
        )

        if x.requires_grad or return_id_experts: 
            auxiliary_loss, balance_loss, router_z_loss = self.combine_loss(
                selected_experts=gate_selected_experts,
                gate_softmax=gate_softmax,
                gate_logits=gate_logits,

            )
            infor_aux = {
                "balance_loss": balance_loss.clone().detach(),
                "router_z_loss": router_z_loss.clone().detach(),
            }
            # selected_experts = gate_selected_experts
            # self.log_metrics['weights'] = gate_weights
            # self.log_metrics['balance_loss'] = balance_loss.item()
            # self.log_metrics['router_z_loss'] = router_z_loss.item()
            # self.log_metrics['gate_softmax'] = gate_softmax
            # self.log_metrics['selected_experts'] = selected_experts
            # infor_aux['selected_experts'] = selected_experts
            # self.log_metrics['balance_loss'] = self.balanceloss(selected_experts=gate_selected_experts, gate_softmax=gate_softmax).item()
        return output, auxiliary_loss, None, infor_aux 

