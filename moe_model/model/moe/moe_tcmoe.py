
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce, pack, unpack
from einops import reduce
import copy

from .register import register_moe
from .moe import MoeLayer


@register_moe("smoe_tcmoe")
class TCMoELayer(MoeLayer):
    def __init__(self, in_embed_dim=768, out_embed_dim=768, num_of_experts=4, num_selected=2, expert=None, args=None, reward_factor=1e-5):
        super().__init__(
            in_embed_dim=in_embed_dim,
            out_embed_dim=out_embed_dim,
            num_of_experts=num_of_experts,
            num_selected=num_selected,
            expert=expert,
            args=args,
        )

        self.num_null = num_selected
        self.router_dim = 2 * self.num_of_experts + self.num_null

        # extend the gate to the router_dim
        self.gate = nn.Linear(self.in_embed_dim, self.router_dim, bias=False)

        # reward factor for null-slot
        self.reward_factor = reward_factor
        self.init_gate_weights()
        self.init_expert_weights()


    def topk_expert(self, gate_logits):
        """
        Apply bias during activation and use sigmoid, then take the top-K heads.
        Matches the sel_activation logic from original implementation.
        """
        # Apply biases before activation to match original implementation
        sel = gate_logits.clone()
        # Negative experts (indices n_experts to 2*n_experts) get -1.0 bias
        sel[..., self.num_of_experts:2*self.num_of_experts] = sel[..., self.num_of_experts:2*self.num_of_experts] - 1.0
        # Null experts (indices 2*n_experts to end) get -10.0 bias  
        sel[..., 2*self.num_of_experts:] = sel[..., 2*self.num_of_experts:] - 10.0
        
        # Apply softmax activation
        gate_softmax =  F.softmax(sel, dim=-1)
        
        # TopK selection
        weights, selected = torch.topk(gate_softmax, self.num_selected)
        
        # # ################ warning #######################
        # weights, selected = torch.topk(gate_softmax, self.num_selected+1)
        
        # weights = weights[:, :, : self.num_selected]
        # selected = selected[:, :, -self.num_selected:]
        
        # 1st +, 2nd -
        # 2nd +, 3st -
        
        # Weight normalization matching original TCMoE implementation
        # Calculate weight sum for activated non-null experts ONLY from selected experts
        weights_from_non_null_experts = weights * (selected < 2 * self.num_of_experts).float()
        weightsum_from_non_null_experts = weights_from_non_null_experts.sum(dim=-1, keepdim=True)
        
        # Calculate weight sum for ALL null experts (since all are conceptually activated)
        weightsum_from_null_experts = gate_softmax[..., 2 * self.num_of_experts:].sum(dim=-1, keepdim=True)
        
        # Normalize ALL gates by the combined weight sum (not just selected ones)
        weightsum = weightsum_from_non_null_experts + weightsum_from_null_experts
        eps = 1e-6
        weightsum = torch.clamp_min(weightsum + 1e-20, eps)

        gates = gate_softmax / weightsum
        
        # Re-gather the normalized weights for selected experts
        weights = torch.gather(gates, -1, selected)
        
        return weights, selected, gate_softmax


    def compute_moe(self, selected_experts, weights, results, x, expert_outputs=None, return_topk_outputs=False):
        # Map ternary indices to (base_expert, sign) & skip null experts
        base_idx = selected_experts % self.num_of_experts
        sign = torch.ones_like(weights)
        neg_mask = (selected_experts >= self.num_of_experts) & (selected_experts < 2 * self.num_of_experts)
        null_mask = selected_experts >= 2 * self.num_of_experts
        
        sign[neg_mask] = -1.0
        sign[null_mask] = 0.0  # Zero out null experts

        # map tokens → per-expert lists
        bucket = {}
        for i in range(self.num_of_experts):
            b,t,k = torch.where(base_idx == i)
            bucket[i] = (b,t,k)
        if return_topk_outputs == True:
            expert_outputs_topk = torch.zeros(x.shape[0], x.shape[1], self.num_of_experts, self.out_embed_dim, device=x.device, dtype=x.dtype)
        
        cached = expert_outputs is not None
        for i, expert in enumerate(self.experts):
            b,t,k = bucket[i]
            if b.numel() == 0:
                continue
            out = expert_outputs[i][b,t] if cached else expert(x[b,t])
            w = weights[b,t,k] * sign[b,t,k]
            results[b,t] += w.unsqueeze(-1) * out
            if return_topk_outputs == True:
                expert_outputs_topk[b, t, i] = out
        if return_topk_outputs:
            B, N, D = x.shape
            
            idx_expanded = base_idx.unsqueeze(-1).expand(B, N, base_idx.shape[-1], x.size(-1))
            topk_expert_outputs = torch.gather(expert_outputs_topk, dim=2, index=idx_expanded)
            diver_loss = self.experts_diversity_loss(topk_expert_outputs)
            if x.requires_grad == False: 
                self.log_metrics['diver_loss'] = diver_loss.item()
            else:
                return results, diver_loss
        return results


    def add_perplexity_reg_tcmoe(self, sel_aux, sel_index, bsz, seq_len):
        """
        TCMoE-specific perplexity regularization matching original implementation
        """
        # only compute perplexity for non-null experts
        sel_aux_non_null = sel_aux[..., :2 * self.num_of_experts]
        sel_aux_null = sel_aux[..., 2 * self.num_of_experts:]
        
        # null mask
        null_mask = sel_index >= 2 * self.num_of_experts
        
        # Reward loss for null experts (matching original implementation)
        if self.reward_factor > 0 and self.training:
            null_prob = sel_aux_null.sum() / (bsz * seq_len)
            reward_loss = -self.reward_factor * null_prob
        else:
            reward_loss = 0.0

        # Load balancing loss for non-null experts only
        if hasattr(self.args, 'balance_loss_coef') and self.args.balance_loss_coef > 0:
            # Normalize probabilities for non-null experts
            p_norm = sel_aux_non_null / (sel_aux_non_null.sum(-1, keepdim=True) + 1e-20)
            
            # Calculate token assignment fractions for non-null experts only
            # Map ternary indices to base expert indices for counting
            base_expert_idx = sel_index % self.num_of_experts
            non_null_mask = sel_index < 2 * self.num_of_experts
            
            # Create one-hot for non-null expert assignments
            aux_loss = torch.zeros(bsz, self.num_of_experts, device=sel_aux.device)
            valid_assignments = base_expert_idx[non_null_mask]
            batch_indices = torch.arange(bsz, device=sel_aux.device).unsqueeze(1).unsqueeze(2).expand(-1, seq_len, self.num_selected)[non_null_mask]
            
            if valid_assignments.numel() > 0:
                aux_loss.scatter_add_(1, valid_assignments.unsqueeze(0).expand(bsz, -1), 
                                    torch.ones_like(valid_assignments).float().unsqueeze(0).expand(bsz, -1))
            
            # Normalize by total non-null tokens per batch
            total_non_null_tokens = non_null_mask.sum(dim=(1, 2)).float()  # [B]
            aux_loss = aux_loss / (total_non_null_tokens.unsqueeze(1) + 1e-20)
            
            # Load balancing loss: sum over experts of (fraction assigned * average probability)
            # Average probability across non-null selections
            p_avg = p_norm.mean(dim=(1))  # [B, 2*n_experts]
            
            # For base experts, combine positive and negative probabilities
            p_base = p_avg[..., :self.num_of_experts] + p_avg[..., self.num_of_experts:2*self.num_of_experts]
            
            # Auxiliary loss
            load_balance_loss = (aux_loss * p_base).sum(dim=1).mean()
            balance_loss = self.args.balance_loss_coef * load_balance_loss
        else:
            balance_loss = 0.0
        
        return balance_loss + reward_loss, balance_loss
    

    def combine_loss(self, selected_experts, gate_softmax, gate_logits):
        bsz, seq_len = gate_logits.shape[:2]
        
        # Use TCMoE-specific loss computation
        auxiliary_loss, balance_loss = self.add_perplexity_reg_tcmoe(
            gate_softmax, selected_experts, bsz, seq_len
        )
        
        # Add router z-loss if specified
        if hasattr(self.args, 'router_z_loss_coef') and self.args.router_z_loss_coef > 0:
            router_z_loss = self.zloss(gate_logits, gate_softmax)
            auxiliary_loss += router_z_loss * self.args.router_z_loss_coef
        
        return auxiliary_loss, balance_loss, router_z_loss


    def forward(self, x, return_id_experts=False, is_vision=False):
        # 1. routing
        gate_logits = self.gate(x)
        weights, selected, gate_softmax = self.topk_expert(gate_logits)

        output = torch.zeros(
            x.shape[0], x.shape[1], self.out_embed_dim,
            device=x.device, dtype=x.dtype
        )
        output = self.compute_moe(selected, weights, output, x, return_topk_outputs=False)
        aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        balance_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        infor_aux = {}
        if self.gate.weight.requires_grad:
            aux_loss, balance_loss, router_z_loss = self.combine_loss(selected, gate_softmax, gate_logits)
            infor_aux = {
                    "balance_loss": balance_loss.clone().detach(),
                    "router_z_loss": router_z_loss.clone().detach()
                }
        if self.gate.weight.requires_grad==False and return_id_experts == True:
            aux_loss, balance_loss, router_z_loss = self.combine_loss(selected, gate_softmax, gate_logits)
            self.log_metrics['weights'] = weights
            self.log_metrics['balance_loss'] = balance_loss.item()
            self.log_metrics['router_z_loss'] = router_z_loss.item()
            self.log_metrics['gate_softmax'] = gate_softmax
            self.log_metrics['selected_experts'] = selected
            self.log_metrics['router_magine'] = weights[:, :, 0] - weights[:, :, 1]

        return output, aux_loss, None, infor_aux