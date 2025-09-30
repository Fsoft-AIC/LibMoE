from moe_model.model.moe.register import register_moe
from moe_model.model.moe.moe import MoeLayer
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist

@register_moe("competesmoev17")
class CompeteSMoEv17(MoeLayer):
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
        # if self.training:
        self.register_buffer('prob_flips', torch.zeros(8316))
        # self.gate = nn.Linear(in_embed_dim, num_of_experts, bias=True)
        # self.layernorm = nn.LayerNorm(num_of_experts, eps=1e-8)
        # self.sigmoid = nn.Sigmoid()
    def set_total_steps(self, step):
        """
        This function sets the total number of training steps (step) and creates a flip schedule
        of size equal to total_steps. Specifically:
        - Warm-up region [0 .. step_warm-1] will always have `False` (i.e., no competition).
        - The region [step_warm .. total_steps-1] will contain 'flip_needed' competition steps
            that are distributed among early, middle, and late portions.

        The logic is as follows:
        1) We calculate step_warm = int(warm_up * step).
        2) We define flip_steps = total_steps - step_warm (the portion where flips can occur).
        3) We compute flip_needed = int(round(flip_steps * rate_flip)) as the total number of
            competition steps we want.
        4) We create a boolean tensor prob_flips of size total_steps, initialized to False.
            In the interval [step_warm .. total_steps-1], we randomly select flip_needed positions
            according to early/middle/late sub-ratios (steps and competition steps).
        5) If we are running in distributed mode, we broadcast this tensor from rank 0 to other ranks.
        6) In the forward pass, we simply check self.prob_flips[current_step] to decide whether
            to activate competition (True) or SMoE (False).

        Args:
            step (int): The total number of training steps.
        """

        self.total_steps = step
        self.step_warm = int(self.warm_up * step)  # Number of warm-up steps
        flip_steps = self.total_steps - self.step_warm
        if flip_steps <= 0:
            raise ValueError("self.total_steps - self.step_warm must be greater than 0.")

        # The total number of competition steps we want
        flip_needed = int(round(flip_steps * self.rate_flip))  # e.g. 0.07 => ~553

        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if rank == 0:
            # 1) Create a boolean tensor of size total_steps, initialized to False
            prob_flips = torch.zeros(self.total_steps, dtype=torch.bool, device=device)

            # We only assign True in the range [step_warm .. total_steps-1].
            # We'll treat [0..flip_steps-1] as 'relative indices' and then offset by step_warm
            # when placing them into prob_flips.

            # 2) Divide flip_steps into early/middle/late
            early_ratio, middle_ratio = 0.30, 0.40
            late_ratio = 1.0 - early_ratio - middle_ratio
            early_steps = int(round(flip_steps * early_ratio))
            middle_steps = int(round(flip_steps * middle_ratio))
            late_steps = flip_steps - early_steps - middle_steps

            # 3) Distribute flip_needed into e + m + l
            e_ratio, m_ratio, l_ratio = 0.5, 0.0, 0.5
            e = int(round(flip_needed * e_ratio))
            m = int(round(flip_needed * m_ratio))
            l = flip_needed - e - m  # leftover for late

            # -- Early region: [0 .. early_steps-1] (relative indices)
            if early_steps > 0 and e > 0:
                idx_early = torch.randperm(early_steps, device=device)[:e]
                # Shift by step_warm to get absolute indices
                prob_flips[idx_early + self.step_warm] = True

            # -- Middle region: [early_steps .. early_steps + middle_steps - 1]
            start_middle = early_steps
            if middle_steps > 0 and m > 0:
                idx_middle_rel = torch.randperm(middle_steps, device=device)[:m]
                idx_middle_abs = idx_middle_rel + (self.step_warm + start_middle)
                prob_flips[idx_middle_abs] = True

            # -- Late region: [early_steps + middle_steps .. flip_steps - 1]
            start_late = early_steps + middle_steps
            if late_steps > 0 and l > 0:
                idx_late_rel = torch.randperm(late_steps, device=device)[:l]
                idx_late_abs = idx_late_rel + (self.step_warm + start_late)
                prob_flips[idx_late_abs] = True

        else:
            # rank != 0 => create empty tensor, then receive broadcast
            prob_flips = torch.empty(self.total_steps, dtype=torch.bool, device=device)

        # Broadcast to other ranks if in distributed mode
        if world_size > 1:
            dist.broadcast(prob_flips, src=0)

        # Check and store
        count_true = prob_flips.sum().item()
        ratio_true = count_true / self.total_steps  # ratio of True compared to total steps
        ratio_true_flipregion = count_true / flip_steps if flip_steps > 0 else 0.0

        self.prob_flips = prob_flips
        self.is_prob_flips = False

        if rank == 0:
            print(f"total_steps                = {self.total_steps}")
            print(f"step_warm                  = {self.step_warm}")
            print(f"flip_steps                 = {flip_steps} (warm-up excluded)")
            print(f"flip_needed                = {flip_needed}, actually used = {count_true}")
            print(f"ratio flip/all_steps       = {ratio_true:.4f}")
            print(f"ratio flip/flip_steps      = {ratio_true_flipregion:.4f}")

            print(f"early_steps                = {early_steps}, middle_steps = {middle_steps}, late_steps = {late_steps}")
            print(f"Competition steps (early)  = {e}")
            print(f"Competition steps (middle) = {m}")
            print(f"Competition steps (late)   = {l}")
            print(f"Sum(e + m + l)            = {e+m+l}\n")



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
            affinity_scores[:, :, i] = torch.mean(F.elu(out_i, alpha = 0.75), dim=-1)
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
        # affinity_softmax, gate_softmax, affinity_logits = affinity_softmax.to(torch.float32), gate_softmax.to(torch.float32), affinity_logits.to(torch.float32)
        routerloss = self.router_loss(
            gate_softmax=gate_softmax,
            affinity_softmax=affinity_softmax
        )
        balance_loss = self.balanceloss(selected_experts=selected_experts, gate_softmax=affinity_softmax)
        loss = routerloss * self.args.balance_loss_coef + balance_loss * self.args.balance_loss_coef
        return loss, balance_loss
    # def combine_loss(self, selected_experts, gate_softmax, gate_logits):
    #     # compute balance loss
    #     balance_loss = self.balanceloss(selected_experts=selected_experts, gate_softmax=gate_softmax)
    #     # compute router_z_loss
    #     router_z_loss = self.zloss(gate_logits, gate_softmax)

    #     auxiliary_loss = balance_loss * self.args.balance_loss_coef 
    #     return auxiliary_loss, balance_loss
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

        auxiliary_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        balance_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # Decide whether to use the competition policy based on `rate_flip`
        if x.requires_grad and self.current_steps >= self.step_warm and self.prob_flips[self.current_steps].item() == 1:
            # print("competition policy for expert selection")
            # Use competition policy for expert selection
            affinity_weights, affinity_selected_experts, affinity_softmax, affinity_logits, expert_outputs = self.competition_policy(x)
        
            routerloss = self.router_loss(
                gate_softmax=gate_softmax,
                affinity_softmax=affinity_softmax.detach()
            )
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