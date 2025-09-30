from moe_model.model.moe.register import register_moe
from moe_model.model.moe.moe import MoeLayer
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
import pandas as pd
@register_moe("competesmoev31")
class CompeteSMoEv31(MoeLayer):
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
        self.register_buffer('prob_flips', torch.zeros(15801))
        self.moe_name = "competesmoe"
        self.init_gate_weights()
    def set_total_steps(self, total_steps, id_layer, prob_flips_final):
        
        assert id_layer is not None, "You must setup id layer is not None"
        assert prob_flips_final is not None, "You must setup prob_flips_final is not None"
        """
        Sets up the total steps for the layer and creates a balanced candidate tensor
        for the current layer. The candidate tensor is adjusted based on the cumulative
        frequency from previous layers to ensure that the threshold is not exceeded,
        and then broadcast across distributed processes.

        Args:
            id_layer (int): Identifier for the current layer.

        Returns:
            dict: Updated prob_flips_final containing candidate tensors for all layers.
        """
        # if self.training == False: return
        # Compute warm-up steps and determine the number of flip steps.
        self.total_steps = total_steps
        self.step_warm = int(self.warm_up * self.total_steps)
        flip_steps = self.total_steps - self.step_warm
        self.flip_steps = flip_steps

        if flip_steps <= 0:
            raise ValueError("self.total_steps - self.step_warm must be greater than 0.")

        # Determine distributed rank and world size.
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        # Set up the device.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def create_balanced_flip_current(cum_frequency):
            """
            Creates a boolean tensor for the current layer with shape [flip_steps].
            For each candidate position, if the random probability (based on self.rate_flip)
            is met but the cumulative frequency (from previous layers plus the current layer)
            would exceed self.max_compete_in_iter, the candidate is shifted left or right
            to find a valid position.

            Args:
                cum_frequency (Tensor): A tensor of shape [flip_steps] containing the cumulative
                                        count of True values from previous layers.

            Returns:
                Tensor: A boolean tensor indicating candidate flips for the current layer.
            """
            candidate_current = [False] * flip_steps  # Initialize candidates.
            freq_updated = cum_frequency.clone()        # Copy cumulative frequency for updates.

            for i in range(flip_steps):
                if torch.rand(1, device=device).item() < self.rate_flip:
                    if freq_updated[i] < self.args.max_compete_in_iter:
                        candidate_current[i] = True
                        freq_updated[i] += 1
                    else:
                        found = False
                        # Try shifting to the left.
                        for j in range(i - 1, -1, -1):
                            if (freq_updated[j] < self.args.max_compete_in_iter) and (not candidate_current[j]):
                                candidate_current[j] = True
                                freq_updated[j] += 1
                                found = True
                                break
                        # If left shift fails, try shifting to the right.
                        if not found:
                            for j in range(i + 1, flip_steps):
                                if (freq_updated[j] < self.args.max_compete_in_iter) and (not candidate_current[j]):
                                    candidate_current[j] = True
                                    freq_updated[j] += 1
                                    found = True
                                    break
            return torch.tensor(candidate_current, dtype=torch.bool, device=device)

        # Only rank 0 creates the candidate tensor.
        if rank == 0:
            from tqdm import tqdm  # Optional progress display.
            import os

            # Compute cumulative frequency from previous layers.
            if prob_flips_final:
                frequency_on_compete = torch.zeros(flip_steps, dtype=torch.int, device=device)
                for _, v in prob_flips_final.items():
                    frequency_on_compete += v.int()
            else:
                frequency_on_compete = torch.zeros(flip_steps, dtype=torch.int, device=device)
                os.environ["start_max"] = '1'

            probs_current = create_balanced_flip_current(frequency_on_compete)
        else:
            # Other ranks create an empty tensor to receive the broadcast.
            probs_current = torch.empty(flip_steps, dtype=torch.bool, device=device)

        # Broadcast the candidate tensor to all processes if in distributed mode.
        if world_size > 1:
            dist.broadcast(probs_current, src=0)

        # Validate the candidate flips.
        count_true = probs_current.sum().item()
        count_false = flip_steps - count_true
        ratio_true = count_true / flip_steps
        ratio_false = count_false / flip_steps

        # if ratio_true == 0.0 or ratio_false == 0.0:
        #     raise ValueError("Invalid ratio of True or False in candidate flips.")

        # Assign the final candidate tensor for the current layer only once.
        prob_flips_final[id_layer] = probs_current
        self.prob_flips = probs_current
        if rank == 0:
            print(f"Updated prob_flips_final keys: {list(prob_flips_final.keys())}")
            print(f"\nCompute Competition Rate (Layer {id_layer}): {ratio_true}")
            print(f"Compute Router Policy Rate: {ratio_false}")
            print(f"Warm-up Steps: {self.step_warm}\n")

        self.is_prob_flips = False
        return prob_flips_final
    def set_current_steps(self, step):
        self.current_steps = step

    def sigmoid(self, x):
        return 2 / (1 + torch.exp(-x))

    def experts_diversity_loss(self, expert_outputs):
        """
        expert_outputs: Tensor shape [B, N, K, D]
            - B: batch size
            - N: sequence length
            - K: number of selected experts
            - D: dimension of each expert output

        Mục tiêu: phạt khi các expert outputs 'quá giống nhau'.
        Ta sẽ tính độ tương đồng cos trung bình giữa mọi cặp (i, j) trong K experts, rồi lấy mean.
        """
        expert_outputs = expert_outputs.to(torch.float32)
        B, N, K, D = expert_outputs.shape

        # Bước 1: Chuẩn hoá (L2-normalize) theo chiều D để tính Cosine Similarity
        # Shape sau chuẩn hoá vẫn là [B, N, K, D]
        normalized = F.normalize(expert_outputs, p=2, dim=-1)

        # Bước 2: Đưa (B, N) về 1 batch lớn để dễ tính bmm
        # Ta reshape thành [B*N, K, D]
        normalized_reshape = normalized.view(B*N, K, D)  # => [B*N, K, D]

        # Bước 3: Tính ma trận similarity bằng bmm:
        # [B*N, K, D] x [B*N, D, K] -> [B*N, K, K]
        similarity_matrix = torch.bmm(
            normalized_reshape, 
            normalized_reshape.transpose(1, 2)
        )  # => [B*N, K, K]

        # Bước 4: Loại bỏ độ tương đồng với chính nó (đường chéo)
        # identity = [K, K], shape broadcast được cho [B*N, K, K]
        mask = 1 - torch.eye(K, device=expert_outputs.device)
        similarity_matrix = similarity_matrix * mask

        # Bước 5: Tính trung bình trên tất cả các batch, token, và cặp expert
        # similarity_matrix có shape [B*N, K, K]. Số phần tử hợp lệ = B*N * K * (K-1)
        loss = similarity_matrix.mean()

        return loss
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
        # with torch.no_grad():
        # Calculate affinity scores based on the norm of each expert's output
        expert_outputs = []
        for i in range(self.num_of_experts):
            out_i = self.experts[i](x)
            affinity_scores[:, :, i] = torch.mean(F.softplus(out_i), dim=-1)
            expert_outputs.append(out_i.unsqueeze(2)) 

        expert_outputs = torch.cat(expert_outputs, dim=2)
        # Compute softmax of the affinity scores
        affinity_softmax = F.softmax(affinity_scores, dim=-1, dtype=torch.float32)
        # Select top experts based on affinity scores
        weights, selected_experts = torch.topk(affinity_scores, self.num_selected)
        weights = weights / torch.sum(weights, dim=-1, keepdim=True).to(x.dtype)
        idx_expanded = selected_experts.unsqueeze(-1).expand(B, N, self.num_selected, expert_outputs.size(-1))
        # just get output in topk 
        topk_expert_outputs = torch.gather(expert_outputs, dim=2, index=idx_expanded)
        return weights, selected_experts, affinity_softmax, affinity_scores, topk_expert_outputs

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
    

    def combine_competition_loss(self, selected_experts, affinity_softmax, gate_softmax, affinity_logits):
        
        routerloss = self.router_loss(
            gate_softmax=gate_softmax,
            affinity_softmax=affinity_softmax
        )
        balance_loss = self.balanceloss(selected_experts=selected_experts, gate_softmax=affinity_softmax)
        loss = routerloss * self.args.balance_loss_coef + balance_loss * self.args.balance_loss_coef
        return loss, balance_loss
 
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
        output = torch.zeros(x.shape[0], x.shape[1], self.out_embed_dim, device=x.device, dtype=x.dtype)

        # Normal gating for expert selection
        gate_weights, gate_selected_experts, gate_softmax, gate_logits = self.router_policy(x)
        # breakpoint()
        auxiliary_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        balance_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # Decide whether to use the competition policy based on `rate_flip`
        if x.requires_grad and self.current_steps >= self.step_warm and self.prob_flips[self.current_steps - self.step_warm].item() == 1:
            # print("competition policy for expert selection")
            # Use competition policy for expert selection
            affinity_weights, affinity_selected_experts, affinity_softmax, affinity_logits, expert_outputs = self.competition_policy(x)
            gate_softmax_topk = torch.gather(gate_softmax, dim=-1, index=affinity_selected_experts)
            affinity_softmax_topk = torch.gather(affinity_softmax, dim=-1, index=affinity_selected_experts)
                
            routerloss = self.router_loss(
                gate_softmax=gate_softmax,
                affinity_softmax=affinity_softmax.detach()
            ) + self.router_loss(
                    affinity_softmax=affinity_softmax_topk.detach(), 
                    gate_softmax=gate_softmax_topk
                    
                ) * self.args.router_theta 
            
            diversity_loss = self.experts_diversity_loss(expert_outputs=expert_outputs)
            
            balance_loss = self.balanceloss(selected_experts=affinity_selected_experts, gate_softmax=affinity_softmax)
            
            auxiliary_loss = routerloss * self.args.router_loss_coef  + diversity_loss * (self.args.balance_loss_coef / 2) + balance_loss * (self.args.balance_loss_coef / 2) 
            
            # Perform MoE computation using competition-selected experts
            output = self.compute_moe(
                selected_experts=affinity_selected_experts,
                weights=affinity_weights,
                results=output,
                x=x            
            )
            selected_experts = affinity_selected_experts
        else:
            # Perform MoE computation using gate-selected experts
            output = self.compute_moe(
                weights=gate_weights,
                selected_experts=gate_selected_experts,
                results=output,
                x=x
            )
            if x.requires_grad: 
                auxiliary_loss, balance_loss = self.combine_loss(
                    selected_experts=gate_selected_experts,
                    gate_softmax=gate_softmax,
                    gate_logits=gate_logits
                )
            selected_experts = gate_selected_experts
        if return_id_experts:
            return output, auxiliary_loss, selected_experts
        else:
            return output, auxiliary_loss, None, balance_loss 
