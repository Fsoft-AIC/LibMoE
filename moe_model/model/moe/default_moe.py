
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce, pack, unpack
from einops import reduce
import copy

from .register import register_moe
from .moe import MoeLayer
@register_moe("default_moe")
class DefaultMoe(MoeLayer):
    def __init__(self, in_embed_dim=768, out_embed_dim=768, num_of_experts=4, num_selected=2, expert=None, args=None):
        super().__init__(in_embed_dim, out_embed_dim, num_of_experts, num_selected, expert, args)
        
        self.in_embed_dim = in_embed_dim
        self.out_embed_dim = out_embed_dim
        self.num_of_experts = num_of_experts
        self.num_selected = num_selected
        self.expert = expert
        self.args = args
        
        self.register_buffer(
            "default_vector",
            torch.zeros(num_of_experts, out_embed_dim, device=torch.cuda.current_device())
        )
        # EMA coefficient – closer to 1 → slower updates.
        self.vector_beta = 0.95
        self.init_gate_weights()
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
        ema_updates = torch.zeros_like(self.default_vector)
        is_expert = expert_outputs is not None
        for i in range(self.num_of_experts):

            expert = self.experts[i]
       
            batch_idx, token_idx, topk_idx = infor_experts[i]
            if batch_idx.numel() == 0 : continue
            if is_expert:
                out_exp = expert_outputs[i][batch_idx, token_idx]
            else:
                out_exp = expert(x[batch_idx, token_idx])

            # -------- Default-vector EMA update --------
            if self.gate.weight.requires_grad:
        
                mean_out = out_exp.mean(dim=0).detach().to(self.default_vector.dtype)
                # Accumulate the (1 - beta) * new_data part
                ema_updates[i] = (1 - self.vector_beta) * mean_out
    
            results[batch_idx, token_idx] += weights[batch_idx, token_idx, topk_idx].unsqueeze(0).T * out_exp
        if self.gate.weight.requires_grad:
            with torch.no_grad():
                self.default_vector = self.default_vector * self.vector_beta + ema_updates
        return results
    def forward(self, x, return_id_experts = False, is_vision = False, out_gate_prev = None):
        gate_logits = self.gate(x)
        weights, selected_experts, gate_softmax = self.topk_expert(gate_logits=gate_logits)
        weights = weights / torch.sum(weights, dim=-1, keepdim=True).to(x.dtype)
        output = torch.zeros(x.shape[0], x.shape[1], self.out_embed_dim, device=x.device, dtype=x.dtype)
        output = self.compute_moe(selected_experts, weights, output, x, return_topk_outputs=True)
        # -----------------------------------------------------
        # Add contribution from *unselected* experts via default vector
        # -----------------------------------------------------
        # gate_softmax : [B, N, E]
        # selected_experts: [B, N, K]
        # Create a binary mask that is 1 at positions of selected experts
        mask = torch.zeros_like(gate_softmax).scatter(2, selected_experts, 1.0)
        # Keep scores only for *unselected* experts (avoid in-place op that breaks autograd)
        masked_scores = gate_softmax * (1.0 - mask)
        # default_vector: [E, D]  →  contribution: [B, N, D]
        default_out = masked_scores.to(output.dtype) @ self.default_vector  # (B,N,D)
        # k = 0.5
        # dist_origin = torch.norm(output)
        # L = None
        # for i in range(0, 10):      
        #     output_sub = k * output + (1.0 - k) * default_out
        #     dis = torch.norm(output_sub)
        #     if L is None:
        #         L = abs(dis - dist_origin)
        #     else:
        #         if dis < L:
        #             L = abs(dis - dist_origin)
        #             k_best = k          
        #     k += 0.1
        # k: 0.5, output: 256.0, default_out: 1.46875
        # because we use sparse upcycling, we need scale it about 1.0 rate
        k_best = 0.9
        output = k_best * output + (1.0 - k_best) * default_out
        # print(f"k: {k_best}, output: {output.norm()}, default_out: {default_out.norm()}")
        # breakpoint()
        # output = output*0.9 + default_out*0.1
        
        aux_loss, balance_loss, router_z_loss = self.combine_loss(selected_experts, gate_softmax, gate_logits)
        infor_aux = {
                "balance_loss": balance_loss.clone().detach(),
                "router_z_loss": router_z_loss.clone().detach()
            }

        return output, aux_loss, None, infor_aux
    