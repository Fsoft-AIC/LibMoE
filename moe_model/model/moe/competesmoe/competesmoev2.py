from moe_model.model.moe.register import register_moe
from moe_model.model.moe.moe import MoeLayer
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn

@register_moe("competesmoev2")
class CompeteSMoEv2(MoeLayer):
    def __init__(self, in_embed_dim=768, out_embed_dim=768, num_of_experts=4, num_selected=2, expert=None, args=None):
        super().__init__(in_embed_dim, out_embed_dim, num_of_experts, num_selected, expert, args)

        if args is None or not hasattr(args, 'rate_flip'):
            raise ValueError("The 'args' parameter must have the attribute 'rate_flip'.")
        if not hasattr(args, 'warm_up'):
            raise ValueError("The 'args' parameter must include 'warm_up'.")
        self.warm_up = args.warm_up # warm up expert with smoe 

        self.rate_flip = args.rate_flip
        # self.aux_loss["router_loss"] = self.router_loss
        self.total_steps = None
        self.current_steps = 0
        self.step_warm = None
        self.sigmoid = nn.Sigmoid()
        self.rate_compete = args.rate_compete 
        # self.prob_flips = torch.tensor([])
        
        
        # # breakpoint()
    def init_luna(self):
        for i in tqdm(range(self.num_of_experts), desc="Initialization Luna"):
            # Lấy trọng số và bias ban đầu
            fc2_weight = list(self.experts[i]._modules.values())[-1].weight
            fc2_bias = list(self.experts[i]._modules.values())[-1].bias
            # breakpoint()
            # Tạo trọng số và bias mới
            mean_weight = fc2_weight.mean(dim=0).unsqueeze(0)
            mean_bias = fc2_bias.mean(dim=0).unsqueeze(0)
            new_fc2_weight = torch.cat([mean_weight, fc2_weight.detach()], dim=0)
            new_fc2_bias = torch.cat([mean_bias, fc2_bias.detach()])

            # Tạo lớp fc2 mới
            new_fc2 = torch.nn.Linear(fc2_weight.shape[1], fc2_weight.shape[0] + 1, bias=True)
            
            # Gán trọng số và bias mới
            with torch.no_grad():
                new_fc2.weight[:] = new_fc2_weight
                new_fc2.bias[:] = new_fc2_bias

            name_last_layer  = list(self.experts[i]._modules.keys())[-1]
            # self.experts[i][-1] = new_fc2
            self.experts[i]._modules[name_last_layer] = new_fc2
    def set_total_steps(self, step):
        self.total_steps = step
        self.step_warm = int(self.warm_up * step)
        # make sure that completesmoe will train both competition and router mode
        # we make list probs flip coin to sure that on gpus has activate same mode
        # if self.prob_flips.shape[0] > 0: return 
        self.register_buffer('prob_flips', torch.tensor([]))
        self.register_buffer('is_compete', torch.tensor([]))
        is_compete = []
        for i in range(self.step_warm):
            if i < self.step_warm * self.rate_compete:
                is_compete.append(True)
            else:
                if torch.rand(1).item() < 0.7:
                    is_compete.append(True)
                else:
                    is_compete.append(False)
        self.is_compete = torch.tensor(is_compete, dtype=torch.bool)
        count_true = sum(is_compete)
        print(f"\nrate warm compute competition: {(self.step_warm - count_true) / self.step_warm} \n step warm smoe: {count_true / self.step_warm }\n")

        while True:
            prob_flips = [torch.rand(1).item() < self.rate_flip for _ in range(self.total_steps - self.step_warm)]
            count_true = sum(prob_flips)
            count_false = (self.total_steps - self.step_warm) - count_true
            ratio_true = count_true / (self.total_steps - self.step_warm)
            ratio_false = count_false / (self.total_steps - self.step_warm)

            if ratio_true > 0.0 and ratio_false > 0.0:
                self.prob_flips = torch.tensor(prob_flips, dtype=torch.bool)
                break
        print(f"\nrate compute competition: {ratio_true}\nrate compute router policy: {ratio_false}\n step warm: {self.step_warm }\n")
    def set_current_steps(self, step):
        self.current_steps = step
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
            affinity_scores[:, :, i] = self.experts[i](x)[:, : , 0]
        # Compute softmax of the affinity scores
        affinity_softmax = F.softmax(affinity_scores, dim=-1, dtype=x.dtype)

        # Select top experts based on affinity scores
        weights, selected_experts = torch.topk(affinity_scores, self.num_selected)
        weights = torch.softmax(weights, dim=-1)

        return weights, selected_experts, affinity_softmax, affinity_scores
    def compute_moe(self, selected_experts, weights, results, x):
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

        for i, expert in enumerate(self.experts):
            batch_idx, token_idx, topk_idx = infor_experts[i]
            results[batch_idx, token_idx] += weights[batch_idx, token_idx, topk_idx].unsqueeze(0).T * expert(x[batch_idx, token_idx])[:, 1:]
        
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
        weights = torch.softmax(weights, dim=-1)

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
            gate_softmax = gate_softmax,
            affinity_softmax = affinity_softmax
        )
        # zloss = self.zloss(gate_logits = affinity_logits)
        balance_loss = self.balanceloss(selected_experts=selected_experts, gate_softmax=affinity_softmax)
        loss = routerloss * self.args.balance_loss_coef + balance_loss *self.args.balance_loss_coef
        return loss        

    def forward(self, x, return_id_experts=False):
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

        
        auxiliary_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        
        # Decide whether to use the competition policy based on `rate_flip`
        if x.requires_grad == True and self.current_steps >= self.step_warm and self.prob_flips[self.current_steps - self.step_warm].item() :
            # normal gating selection experts
            gate_weights, gate_selected_experts, gate_softmax, gate_logits = self.router_policy(x)
            # Use competition policy for expert selection
            affinity_weights, affinity_selected_experts, affinity_softmax, affinity_logits = self.competition_policy(x)
            auxiliary_loss = self.combine_competition_loss(
                affinity_softmax = affinity_softmax, # 
                gate_softmax = gate_softmax,
                affinity_logits = affinity_logits,
                selected_experts=affinity_selected_experts
            )
            # Perform MoE computation using competition-selected experts
            output = self.compute_moe(
                selected_experts=affinity_selected_experts,
                weights=affinity_weights,
                results=output,
                x=x
            )

            selected_experts = affinity_selected_experts
        else:
            if x.requires_grad == False or self.current_steps <= int(self.step_warm * self.rate_compete) or self.current_steps >= self.step_warm or self.is_compete[self.current_steps]:
                # print("warm with smoe")
                # normal gating selection experts
                gate_weights, gate_selected_experts, gate_softmax, gate_logits = self.router_policy(x)

                # Perform MoE computation using gate-selected experts
                output = self.compute_moe(
                    weights=gate_weights,
                    selected_experts=gate_selected_experts,
                    results=output,
                    x=x
                )
                if x.requires_grad == True: 
                    auxiliary_loss = self.combine_loss(
                        selected_experts = gate_selected_experts,
                        gate_softmax = gate_softmax,
                        gate_logits = gate_logits
                    )
                selected_experts = gate_selected_experts
            elif self.current_steps <= self.step_warm :
                affinity_weights, affinity_selected_experts, affinity_softmax, affinity_logits = self.competition_policy(x)
                # Perform MoE computation using competition-selected experts
                output = self.compute_moe(
                    selected_experts=affinity_selected_experts,
                    weights=affinity_weights,
                    results=output,
                    x=x
                )
                if x.requires_grad == True: 
                    auxiliary_loss = self.combine_loss(
                        selected_experts = affinity_selected_experts,
                        gate_softmax = affinity_softmax,
                        gate_logits = affinity_logits
                    )
                selected_experts = affinity_selected_experts

        # breakpoint()
        if return_id_experts:
            
            return output, auxiliary_loss, selected_experts
        else:
            return output, auxiliary_loss, None
