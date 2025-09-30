
import torch
import torch.nn as nn
import torch.nn.functional as F
from moe_model.model.moe.register import register_moe
from moe_model.model.moe.moe import MoeLayer
from .utils import unique_each_row_vectorized, create_table_from_index_and_value
from datetime import datetime

@register_moe("dynamic_moe_luna")
class DynamicMoELuna(MoeLayer):
    
    def __init__(self, in_embed_dim=768, out_embed_dim=768, num_of_experts=4, num_selected=2, expert=None, args=None):
        super().__init__(in_embed_dim, out_embed_dim, num_of_experts, num_selected, expert, args)
        print("Starting with LuNa Dynamic MoE")
        self.gate = nn.Linear(in_embed_dim, num_of_experts, bias=True) 
        
        self.total_steps = None
        self.current_steps = 1
        self.num_current_selected = 1
        self.number_of_previous_tokens = args.number_of_previous_tokens
        self.normalization = args.normalization
        self.dynamic_activate = True
        self.num_selected = 1
        # Get the current time
        # Format the string with day, month, year, hour, minute, and second
        self.name_logs = datetime.now().strftime("`dynamic_`%Y-%m-%dT%H-%M-%S.%f.txt")
        assert self.number_of_previous_tokens <= self.num_of_experts, "The number of preceding tokens must be less than the number of experts. \n num_of_experts: {self.num_of_experts} \n number_of_previous_tokens: {number_of_previous_tokens}"
        self.aux_loss['eloss'] = self.entropy_loss
        # self.aux_loss['eadv_loss'] = self.entropy_adv
        # self.aux_loss['eadv2_loss'] = self.entropy_adv2
        self.aux_loss['routing_diversity_loss_all_tokens'] = self.routing_diversity_loss_all_tokens
        
        print(f"Apply loss 1: {args.loss1}")
        print(f"Apply loss 2: {args.loss2}")
        try:
            self.loss1 = self.aux_loss[args.loss1]
            self.loss2 = self.aux_loss[args.loss2]
        except:
            print("Warning: Not define loss 1 and loss 2")
            self.loss1 = self.aux_loss['balanceloss']
            self.loss2 = self.aux_loss['zloss']
        self.layer_norm = nn.Identity()
        if self.args.normalization:
            print("Apply Normalization")
            self.layer_norm = nn.LayerNorm(in_embed_dim)
    def set_total_steps(self, step):
        self.total_steps = step

    def set_current_steps(self, step):
        self.current_steps = step
    # def strategy_learning(self, ) 
    def set_num_current_selected(self):
        # linear increase 
        # if self.current_steps > self.total_steps*0.2 and self.dynamic_activate == False:
        #     self.dynamic_activate = True
        #     print("Apply Dynamic Expert")
        #     with open("/cm/shared/anonymous_H102/toolkitmoe/log_dynamic.txt", 'a') as file:
        #         file.write("Apply Dynamic Expert\n")
        percent_progress = int((self.current_steps / self.total_steps)*100)
        if self.current_steps > 0 and percent_progress % 10 == 0:
            self.dynamic_activate = not self.dynamic_activate
            # Get the current time with microsecond precision
            current_time = datetime.now()
            # Format the timestamp in a professional log format (ISO 8601 with milliseconds)
            timestamp = current_time.strftime("%Y-%m-%dT%H-%M-%S.%f")[:-3]  # Trim microseconds to milliseconds

            with open(f"/cm/shared/anonymous_H102/toolkitmoe/logs/{self.name_logs}", 'a') as file:
                file.write(f"{timestamp}: Apply Dynamic Expert at {percent_progress}% data: {self.dynamic_activate}\n")

    
    def entropy_loss(self, gate_logits, gate_softmax):
        """
        Computes the entropy loss to penalize uniform distributions after softmax.

        Args:
            gate_logits (torch.Tensor): The logits from the gating network.

        Returns:
            torch.Tensor: The computed entropy loss value.
        """
        # Apply softmax to convert logits to probabilities
        # gate_softmax = F.softmax(gate_logits, dim=-1)
        max_value = torch.max(gate_softmax, dim=-1).values
        # Calculate entropy
        entropy = (1 - max_value) * (-torch.sum(gate_softmax * torch.log(gate_softmax + 1e-12), dim=-1))

        # Return the mean entropy loss (lower entropy means less uniform distribution)
        return entropy.mean()
    def routing_diversity_loss_all_tokens(self, gate_logits, gate_softmax):
        """
        Tính toán loss để khuyến khích sự đa dạng trong phân phối gating của tất cả các token trong một sample.
        
        Args:
            gate_logits (torch.Tensor): Tensor kích thước [B, N, E]
            penalty_weight (float): Trọng số cho loss
            temperature (float): Nhiệt độ cho softmax
        
        Returns:
            torch.Tensor: Giá trị loss
        """
        B, N, E = gate_logits.size()
        
        # Bước 1: Áp dụng softmax để tạo phân phối xác suất gating
        gating_probs = gate_softmax  # Shape: [B, N, E]
        
        # Bước 2: Chuẩn hóa phân phối gating để tính cosine similarity
        gating_norm = F.normalize(gating_probs, p=2, dim=-1)  # Shape: [B, N, E]
        
        # Bước 3: Tính toán ma trận tương đồng cosine giữa các token trong mỗi sample
        # gating_norm: [B, N, E]
        # similarity_matrix sẽ là [B, N, N]
        similarity_matrix = torch.bmm(gating_norm, gating_norm.transpose(1, 2))  # Shape: [B, N, N]
        
        # Bước 4: Loại bỏ self-similarity bằng cách nhân với (1 - ma trận đơn vị)
        # Tạo ma trận đơn vị [B, N, N]
        identity = torch.eye(N, device=gate_logits.device).unsqueeze(0).repeat(B, 1, 1)  # Shape: [B, N, N]
        similarity_matrix = similarity_matrix * (1 - identity)  # Đặt self-similarity = 0
        
        # Bước 5: Tính loss là trung bình của các giá trị tương đồng
        # similarity_matrix có giá trị từ -1 đến 1, nhưng với softmax sẽ chủ yếu từ 0 đến 1
        # Vì vậy, chúng ta có thể sử dụng mean của similarity_matrix
        loss = similarity_matrix.mean()
        
        return loss

    
    def combine_loss(self, selected_experts, gate_softmax, gate_logits):
        
        # Kiểm tra NaN ở gate_softmax
        # if torch.isnan(gate_softmax).any():
        #     print("NaN detected in gate_softmax")
        # if torch.isnan(selected_experts).any():
        #     print("NaN detected in selected_experts")
        # if torch.isnan(gate_logits).any():
        #     print("NaN detected in gate_logits")
        # compute balance loss
        balance_loss = self.loss1(selected_experts=selected_experts, gate_softmax=gate_softmax)
        # compute router_z_loss
        # return balance_loss * self.args.balance_loss_coef gating_logits, gate_softmax
        entropy_loss = self.loss2(gate_logits = gate_logits, gate_softmax = gate_softmax)
        

        # print(f"Entropy: {entropy_loss}")
        # balance_loss = self.balanceloss(selected_experts=selected_experts, gate_softmax=gate_softmax)
        # entropy_loss = self.entropy_adv2(gate_logits = gate_logits, gate_softmax = gate_softmax)
        # zloss = self.zloss(gate_logits=gate_logits, gate_softmax=gate_softmax)
        auxiliary_loss = balance_loss * self.args.balance_loss_coef + entropy_loss * self.args.router_z_loss_coef 
        
        return auxiliary_loss, balance_loss
    def unique_index(self, batch_id, token_id, topk):
        
        # batch_id = torch.flip(batch_id, dims=[-1])
        # token_id = torch.flip(token_id, dims=[-1])
        # topk = torch.flip(topk, dims=[-1])
        # 1. Ghép batch_id và topk vào cùng một tensor
        pairs = torch.stack([batch_id, token_id], dim=-1)  # shape: (N, 2)
        # 2. Lấy các cặp (batch_id, topk) duy nhất bằng torch.unique
        unique_pairs, index = torch.unique(pairs, dim=0, return_inverse=True)
        topk_new = torch.zeros(unique_pairs.shape[0], dtype=topk.dtype, device=topk.device)

        topk_new[index] = topk
        token_id_new = unique_pairs[:, 1]
        batch_id_new = unique_pairs[:, 0]
        return batch_id_new, token_id_new, topk_new
    # def unique_index(self, batch_id, token_id, topk):
    #     """
    #     Giữ lại (batch_id, token_id) xuất hiện đầu tiên trong danh sách gốc.
    #     Nếu cặp (b, t) xuất hiện nhiều lần, ta chỉ lấy topk của lần đầu.
    #     """
    #     # 1) Đảo ngược thứ tự
    #     batch_id_rev = batch_id.flip(0)  # [N] -> [N] (đảo ngược)
    #     token_id_rev = token_id.flip(0)
    #     topk_rev = topk.flip(0)
    #     breakpoint()
    #     # 2) Ghép cặp (b, t) đã đảo
    #     pairs_rev = torch.stack([batch_id_rev, token_id_rev], dim=-1)  # shape: (N, 2)

    #     # 3) unique() + return_inverse => inv_idx
    #     unique_pairs, inv_idx = torch.unique(pairs_rev, dim=0, return_inverse=True)

    #     # 4) Gán topk_new_rev
    #     topk_new_rev = torch.zeros(
    #         unique_pairs.size(0),
    #         dtype=topk.dtype,
    #         device=topk.device
    #     )
    #     # “Ghi đè” 1 lần duy nhất => giữ giá trị cuối của pairs_rev 
    #     # => chính là giá trị *đầu* của pairs gốc
    #     topk_new_rev[inv_idx] = topk_rev

    #     # 5) Tách ra batch_id_new, token_id_new
    #     batch_id_new = unique_pairs[:, 0]
    #     token_id_new = unique_pairs[:, 1]

    #     return batch_id_new, token_id_new, topk_new_rev
    def compute_moe_dynamic(self, gate_logits, selected_experts, weights, results, x):
        
        # with torch.no_grad():
        x_padded = F.pad(selected_experts, pad=(0, 0, self.number_of_previous_tokens-1, 0), mode='constant', value=-1) 
        # stride get before token
        x_unfold = x_padded.unfold(dimension=1, size=self.number_of_previous_tokens, step=1)  
        # flip to expert top 1 up to first place
        x_unfold = torch.flip(x_unfold, dims=[-1])

        selected_experts_stride = x_unfold.squeeze(-1).squeeze(2)  # Kích thước (B, N, L)

        infor_experts = {}
        #make mask to get weight
        mask_selected_experts_stride = torch.zeros(selected_experts_stride.shape, dtype=selected_experts_stride.dtype, device=selected_experts_stride.device)

        for i in range(self.num_of_experts):
            batch_idx, token_idx, topk_idx = torch.where(selected_experts_stride== i)
            # get unique at position same (batch id and token id)
            
            batch_idx_new, token_idx_new, topk_idx_new = self.unique_index(
                token_id=token_idx, 
                batch_id=batch_idx, 
                topk =topk_idx
            )
            
            mask_selected_experts_stride[batch_idx_new, token_idx_new, topk_idx_new] +=1
            infor_experts[i] = [batch_idx_new, token_idx_new, topk_idx_new]
            
        mask_selected_experts_stride = mask_selected_experts_stride > 0
        unique_weight = weights*mask_selected_experts_stride
            
        # print("Indices of [False, False] pairs:", false_false_indices)
        softmax_weight = unique_weight / unique_weight.sum(dim=-1, keepdim=True)
        
        for i, expert in enumerate(self.experts):
            batch_idx, token_idx, topk_idx = infor_experts[i]
            weights_slice = softmax_weight[batch_idx, token_idx, topk_idx].unsqueeze(0).T
            results[batch_idx, token_idx] += weights_slice*expert(x[batch_idx, token_idx])
            # print(torch.isnan(results[batch_idx, token_idx] ).any())
        return results
    
    def forward(self, x, return_id_experts = False, is_vision = False):
        # if torch.isinf(x).any():
        #     print("Inf detected in input x")
        if self.normalization :

            gate_logits = self.gate(self.layer_norm(x))
            # x = x / x.norm(p=2, dim=-1, keepdim=True)
            # gate_logits = self.gate(x / x.norm(p=2, dim=-1, keepdim=True))
        else:
            gate_logits = self.gate(x)

        gate_softmax = F.softmax(gate_logits, dim=-1, dtype=torch.float).to(x.dtype)
        weights, selected_experts_full = torch.topk(gate_softmax, k=self.number_of_previous_tokens, dim=2)
        selected_experts = selected_experts_full[:, :, :1]
        
        
        output = torch.zeros(x.shape[0], x.shape[1], self.out_embed_dim, device=x.device, dtype=x.dtype)

        auxiliary_loss = torch.tensor(0.0, device=x.device)

        balance_loss = torch.tensor(0.0, device=x.device)        
        
        output = self.compute_moe_dynamic(gate_logits = gate_softmax, selected_experts = selected_experts, weights = weights, results = output, x = x)
        if x.requires_grad == True:
            auxiliary_loss, balance_loss = self.combine_loss(selected_experts, gate_softmax, gate_logits)
        if return_id_experts:
            return output, auxiliary_loss, selected_experts
        else: 
            return output, auxiliary_loss, None, balance_loss
