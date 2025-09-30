from .register import register_moe
from .moe import MoeLayer

import torch
import torch.nn as nn
import torch.nn.functional as F
from moe_model.model.multimodal_encoder.siglip_smoe import SiglipMLP

@register_moe("smoe_diversity")
class SMoeDivLayer(MoeLayer):
    def __init__(self, in_embed_dim=768, out_embed_dim=768, num_of_experts=4, num_selected=2, expert=None, args=None):
        super().__init__(in_embed_dim, out_embed_dim, num_of_experts, num_selected, expert, args)
        
        self.log_metrics = {}
        self.is_vision = False
        self.std_gate = getattr(self.args, "std_gate", 0.02)
        self.init_gate_weights(self.std_gate)
        self.init_expert_weights()
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
    def forward(self, x,  return_id_experts = False, is_vision=False):
        self.is_vision = is_vision
        gate_logits = self.gate(x)
        weights, selected_experts, gate_softmax = self.topk_expert(gate_logits=gate_logits)
        # weights, selected_experts, gate_softmax, gate_logits = self.competition_policy(x=x)
        weights = weights / torch.sum(weights, dim=-1, keepdim=True).to(x.dtype)
        output = torch.zeros(x.shape[0], x.shape[1], self.out_embed_dim, device=x.device, dtype=x.dtype)
        # output = self.compute_moe_dynamic(gate_logits = gate_softmax, selected_experts = selected_experts, weights = weights, results = output, x = x)
        output, div_loss = self.compute_moe(selected_experts, weights, output, x, return_topk_outputs=True)
        auxiliary_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        balance_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        infor_aux = {}
        # compute loss
        if x.requires_grad or return_id_experts: 
            auxiliary_loss, balance_loss, router_z_loss = self.combine_loss(selected_experts, gate_softmax, gate_logits)
            infor_aux = {
                "balance_loss": balance_loss.clone().detach(),
                "router_z_loss": router_z_loss.clone().detach(),
                "div_loss": div_loss.clone().detach(),
                
            }
            auxiliary_loss += div_loss*0.005
        if self.gate.weight.requires_grad == False and return_id_experts == True:
            auxiliary_loss, balance_loss, router_z_loss = self.combine_loss(selected_experts, gate_softmax, gate_logits)
            self.log_metrics['weights'] = weights
            self.log_metrics['balance_loss'] = balance_loss.item()
            self.log_metrics['router_z_loss'] = router_z_loss.item()
            self.log_metrics['gate_softmax'] = gate_softmax
            self.log_metrics['selected_experts'] = selected_experts

        return output, auxiliary_loss, None, infor_aux
