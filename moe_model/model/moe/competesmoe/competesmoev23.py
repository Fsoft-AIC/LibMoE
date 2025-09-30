from moe_model.model.moe.register import register_moe
from moe_model.model.moe.moe import MoeLayer
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist

@register_moe("competesmoev23")
class CompeteSMoEv23(MoeLayer):
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
        # self.register_buffer('prob_flips', torch.zeros(7901))
        self.register_buffer('prob_flips', torch.zeros(15801))
        # self.gate = nn.Linear(in_embed_dim, num_of_experts, bias=True)
        # self.layernorm = nn.LayerNorm(num_of_experts, eps=1e-8)
        # self.sigmoid = nn.Sigmoid()
        self.init_gate_weights()
    def set_total_steps(self, step):
        total_sum = self.prob_flips.sum()
        
        self.total_steps = step
        self.step_warm = int(self.warm_up * step)
        flip_steps = self.total_steps - self.step_warm
        if total_sum > 0: return
        if flip_steps <= 0:
            raise ValueError("self.total_steps - self.step_warm must be greater than 0.")

        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        # Determine the current device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if rank == 0:
            # Rank 0 generates prob_flips
            prob_flips = [torch.rand(1, device=device).item() < self.rate_flip for _ in range(flip_steps)]
            prob_flips = torch.tensor(prob_flips, dtype=torch.bool, device=device)
        else:
            # Other ranks prepare an empty tensor to receive data
            prob_flips = torch.empty(flip_steps, dtype=torch.bool, device=device)

        if world_size > 1:
            # All ranks call broadcast
            dist.broadcast(prob_flips, src=0)

        # Check ratio after broadcast
        count_true = prob_flips.sum().item()
        count_false = flip_steps - count_true
        ratio_true = count_true / flip_steps
        ratio_false = count_false / flip_steps

        if ratio_true == 0.0 or ratio_false == 0.0:
            raise ValueError("Invalid ratio of true or false in prob_flips.")

        self.prob_flips = prob_flips
        self.is_prob_flips = False

        print(f"\nRate compute competition: {ratio_true}\nRate compute router policy: {ratio_false}\nStep warm: {self.step_warm}\n")

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
        if x.requires_grad and self.current_steps >= self.step_warm and self.prob_flips[self.current_steps - self.step_warm].item() == 1:
            # print("competition policy for expert selection")
            # Use competition policy for expert selection
            affinity_weights, affinity_selected_experts, affinity_softmax, affinity_logits, expert_outputs = self.competition_policy(x)
        
            routerloss = self.router_loss(
                gate_softmax=F.sigmoid(gate_logits.to(torch.float32)),
                affinity_softmax=F.sigmoid(affinity_logits.to(torch.float32)).detach()
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