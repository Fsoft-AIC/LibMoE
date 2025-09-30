from moe_model.model.moe.register import register_moe
from moe_model.model.moe.moe import MoeLayer
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
import pandas as pd
import copy 
@register_moe("moeskv4")
class MoESKv4(MoeLayer):
    def __init__(self, in_embed_dim=768, out_embed_dim=768, num_of_experts=4, num_selected=2, expert=None, args=None):
        super().__init__(in_embed_dim, out_embed_dim, num_of_experts , num_selected , expert, args)

    
        self.warm_up = args.warm_up  # Warm up expert with SMoE
        self.rate_flip = args.rate_flip
        self.total_steps = None
        self.current_steps = 0
        self.step_warm = None
        self.is_prob_flips = True
        self.hybrid = False
        self.expert_shared = copy.deepcopy(self.experts[0])
        self.register_buffer('prob_flips', torch.zeros(15801))
        self.init_gate_weights()
        self.log_metrics = {}
        self.lamda_shared = 0.5
        self.nb_wakers = 10
        self.id_layer = None
        self.rate_balance = 0.8
        self.max_entropy = torch.log2(torch.tensor(self.num_of_experts))*self.rate_balance
        self.global_balance = torch.zeros((self.num_of_experts), requires_grad=False)
        # self.global_balance = torch.tensor([3, 4, 1])

    def high_outliers(self, x: torch.Tensor):
        '''`
        Get experts with frequency selected at history, sorted by frequency in descending order.
        
        Args:
            x (torch.Tensor): Input tensor of frequencies.
        
        Returns:
            tuple: (idx, x_idx)
                - idx (torch.Tensor): Indices of elements above the mean, sorted by their values in descending order.
                - x_idx (torch.Tensor): Values of x at those indices, sorted accordingly.
        '''
        x = x.float()  # Convert to float32 for computation
        mask = x > 0  # Mask for elements above the mean

        # Get indices where mask is True
        idx = mask.nonzero(as_tuple=True)[0]
        
        if idx.numel() == 0:
            return idx, x[idx]  # Return empty tensors if no elements are above mean
        
        # Get values at those indices
        x_idx = x[idx]
        
        # Sort indices by values in descending order
        sorted_indices = torch.argsort(x_idx, descending=True)
        idx_sorted = idx[sorted_indices]
        x_idx_sorted = x_idx[sorted_indices]
        
        return idx_sorted, x_idx_sorted
    def set_total_steps(self, total_steps):
        self.total_steps = total_steps
    def set_current_steps(self, step):
        self.current_steps = step
    
    def shared_knowlegde(self, x, gate_softmax, output_shared):
        
        share_loss = None
        
        _, selected_experts = torch.topk((-1) * gate_softmax, self.num_of_experts - self.num_selected) # 3, 1, - 1 shared exp 
        for i in range(self.num_of_experts):
            batch_idx, token_idx, topk_idx = torch.where(selected_experts == i)
            if batch_idx.numel() == 0 : continue
            out_exp = self.experts[i](x[batch_idx, token_idx])
             
            if share_loss == None:
                share_loss = F.mse_loss(out_exp, output_shared[batch_idx, token_idx])
            else:
                share_loss += F.mse_loss(out_exp, output_shared[batch_idx, token_idx])
        return share_loss
    import torch

    import torch

    def compute_entropy_score(self, probabilities, normalize=False, eps=1e-10):
        """
        Compute the Shannon entropy score of a probability distribution using PyTorch.
        
        Args:
            probabilities (torch.Tensor): Tensor of probabilities or unnormalized scores.
                                        Shape: (N,) for single distribution or (B, N) for batch.
            normalize (bool): If True, apply softmax to convert scores to probabilities.
            eps (float): Small value to avoid log(0).
        
        Returns:
            torch.Tensor: Entropy score in bits (scalar for single dist, shape (B,) for batch).
            
        Raises:
            ValueError: If probabilities are invalid (e.g., negative, empty, or sum to zero).
        """
        # Ensure input is a tensor
        if not isinstance(probabilities, torch.Tensor):
            probabilities = torch.tensor(probabilities, dtype=torch.float)
        
        # Check for valid input
        if probabilities.numel() == 0:
            raise ValueError("Probability tensor cannot be empty.")
        if torch.any(probabilities < 0):
            raise ValueError("Probabilities cannot be negative.")
        
        probs = probabilities.to(dtype=torch.float32)
    
        if normalize:
            # Chuẩn hóa bằng tổng (thay vì softmax)
            sum_probs = torch.sum(probs, dim=-1, keepdim=True)
            # if sum_probs
            if torch.any(sum_probs == 0):
                return self.max_entropy
                raise ValueError("Không thể chuẩn hóa: tổng bằng 0.")
            probs = probs / sum_probs
        
        # Giới hạn xác suất trong khoảng [eps, 1.0]
        probs = torch.clamp(probs, min=eps, max=1.0)
        
        # Tính entropy: -sum(p * log2(p))
        entropy = -torch.sum(probs * torch.log2(probs), dim=-1)
        
        return entropy.to(probabilities.dtype)
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
        B, N, D = gate_logits.shape
        gate_softmax = F.softmax(gate_logits, dim = -1)
        # hard balancing loss
        e_score = self.compute_entropy_score(self.global_balance, normalize = True)
        
        global_experts = 1 - self.global_balance / ( torch.sum(self.global_balance, dim=-1, keepdim=True).to(x.dtype) + 1e-9)
        d_w = gate_softmax.view(B * N, D).sum(dim=0)
        d_w = d_w / torch.sum(d_w, dim=-1, keepdim=True).to(x.dtype)
        bal_loss = F.mse_loss(d_w, global_experts.to(dtype=x.dtype).to(device=x.device).detach())
        
        # Select experts using top-k gating
        weights, selected_experts = torch.topk(gate_softmax, self.num_selected)
        
        # update global balancing
        flat_experts = selected_experts.view(-1)
        one_hot = torch.nn.functional.one_hot(flat_experts, num_classes=self.num_of_experts)
        counts = one_hot.sum(dim=0)
        self.global_balance = self.global_balance.to(device=x.device) + counts.view(1, -1).view(-1) 
        
        weights = weights / torch.sum(weights, dim=-1, keepdim=True).to(x.dtype)

        return weights, selected_experts, gate_softmax, gate_logits, e_score, bal_loss
    
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
        gate_weights, gate_selected_experts, gate_softmax, gate_logits, e_score, bal_loss = self.router_policy(x)
        auxiliary_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        infor_aux = {}
        output_shared = self.expert_shared(x)
        shared_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
   
        auxiliary_loss = self.zloss(gate_logits, gate_softmax) * self.args.router_z_loss_coef  + bal_loss*0.001
        # + bal_loss * self.args.balance_loss_coef

        # Perform MoE computation using gate-selected experts
        output = self.compute_moe(
            weights=gate_weights,
            selected_experts=gate_selected_experts,
            results=output,
            x=x
        )
        
        # Decide whether to use the competition policy based on `rate_flip`
        if x.requires_grad and (self.current_steps + 1) % (self.nb_wakers + self.id_layer) == 0:
            
            # Use competition policy for expert selection
            shared_loss = self.shared_knowlegde(
                    x = x,
                    output_shared = output_shared.detach(),
                    gate_softmax=gate_softmax                
                )
            auxiliary_loss += shared_loss*0.1

        
        infor_aux = {
            "router_z_loss": auxiliary_loss.clone().detach(),
            "shared_loss": shared_loss,
            "e_score": e_score.to(device=x.device), 
            "bal_loss": bal_loss.clone().detach()
        }
        # if torch.cuda.current_device() == 0:
        #     print(f"Layer {self.id_layer} - {self.global_balance} - e_score: {e_score}")
        output_final = output_shared * 0.5 + output *  0.5
        return output_final, auxiliary_loss, None, infor_aux 
