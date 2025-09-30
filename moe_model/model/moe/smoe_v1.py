from .register import register_moe
from .moe import MoeLayer

import torch
import torch.nn as nn
import torch.nn.functional as F

@register_moe("smoev1")
class smoev1(MoeLayer):
    def __init__(self, in_embed_dim=768, out_embed_dim=768, num_of_experts=4, num_selected=2, expert=None, args=None):
        super().__init__(in_embed_dim, out_embed_dim, num_of_experts, num_selected, expert, args)


    # def router_loss(self, selected_experts):

    #     # breakpoint()
    #     # compress router expert
    #     one_top1 = F.one_hot(selected_experts[:, :, 0].unsqueeze(-1), 
    #                     num_classes=self.num_of_experts).squeeze(2)
        
    #     one_top2 = F.one_hot(selected_experts[:, :, 1].unsqueeze(-1), 
    #                     num_classes=self.num_of_experts).squeeze(2)
    #     k = 1
    #     target = torch.cat((one_top1[:, :k, :], one_top1[:, :-k, :]), dim=1)
    #     one_top2 = one_top2.to(dtype=torch.float32)
    #     target = target.to(dtype=torch.float32)
    #     loss = F.cross_entropy(one_top2, target)

    #     return loss
    def router_loss(self, selected_experts, gate_logits):
        batch_size, seq_len, num_selected = selected_experts.shape
        num_of_experts = gate_logits.size(-1)

        if seq_len <= 1:
            return torch.tensor(0.0, device=gate_logits.device)

        # Lấy chỉ số expert top-1 của token trước đó
        prev_top1 = selected_experts[:, :-1, 0]  # shape: (batch_size, seq_len - 1)

        # Lấy logits của gate tại thời điểm hiện tại (bỏ qua token đầu tiên)
        curr_gate_logits = gate_logits[:, 1:, :]  # shape: (batch_size, seq_len - 1, num_of_experts)

        # Reshape để phù hợp với hàm cross_entropy
        curr_gate_logits = curr_gate_logits.reshape(-1, num_of_experts)
        prev_top1 = prev_top1.reshape(-1)

        # Tính toán cross-entropy loss
        loss = F.cross_entropy(curr_gate_logits, prev_top1)

        return loss

    def combine_loss(self, selected_experts, gate_softmax, gate_logits):
        # compute balance loss
        balance_loss = self.balanceloss(selected_experts=selected_experts, gate_softmax=gate_softmax)
        # compute router_z_loss
        router_loss = self.router_loss(selected_experts, gate_logits)

        auxiliary_loss = balance_loss * self.args.balance_loss_coef + \
            router_loss * self.args.router_z_loss_coef
        
        return auxiliary_loss, router_loss
    def forward(self, x,  return_id_experts = False):
 
        gate_logits = self.gate(x)
        weights, selected_experts, gate_softmax = self.topk_expert(gate_logits=gate_logits)
        

        weights = weights / torch.sum(weights, dim=-1, keepdim=True).to(x.dtype)
        output = torch.zeros(x.shape[0], x.shape[1], self.out_embed_dim, device=x.device, dtype=x.dtype)
        output = self.compute_moe(selected_experts, weights, output, x)
        
        # compute loss
        auxiliary_loss, router_loss = self.combine_loss(selected_experts, gate_softmax, gate_logits)
        if return_id_experts:
            return output, auxiliary_loss, None, router_loss
        else:
            return output, auxiliary_loss, None, router_loss
