
import torch
import torch.nn as nn
import torch.nn.functional as F
from moe_model.model.moe.register import register_moe
from moe_model.model.moe.moe import MoeLayer
from .utils import unique_each_row_vectorized, create_table_from_index_and_value
from datetime import datetime

def build_expert_union(selected_experts: torch.Tensor, num_prev: int):
    B, N = selected_experts.size()
    out = torch.full((B, N, num_prev), fill_value=-1,
                     device=selected_experts.device, dtype=selected_experts.dtype)
    out[:, :, 0] = selected_experts
    for i in range(1, num_prev):
        out[:, i:, i] = selected_experts[:, :-i]
    return out

def row_unique_and_sort(tensor_3d: torch.Tensor, fill_value=-1):
    sorted_tensor, _ = torch.sort(tensor_3d, dim=-1)
    mask = torch.cat([
        torch.ones_like(sorted_tensor[:, :, :1], dtype=torch.bool),
        (sorted_tensor[:, :, 1:] != sorted_tensor[:, :, :-1])
    ], dim=-1)
    unique_tensor = torch.where(mask, sorted_tensor, fill_value)
    unique_tensor, _ = torch.sort(unique_tensor, dim=-1)
    return unique_tensor

@register_moe("dynamic_moe")
class DynamicMoE(MoeLayer):
    
    def __init__(self, in_embed_dim=768, out_embed_dim=768, num_of_experts=4, num_selected=2, expert=None, args=None):
        super().__init__(in_embed_dim, out_embed_dim, num_of_experts, num_selected, expert, args)
        print("Starting with Dynamic MoE")
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
        self.aux_loss['eadv_loss'] = self.entropy_adv
        self.aux_loss['eadv2_loss'] = self.entropy_adv2
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
    def entropy_adv(self, gate_logits, k=2):
        """
        Computes the modified loss, which combines entropy loss with a regularization term.

        Args:
            gate_logits (torch.Tensor): The logits from the gating network.
            k (int): The power for the regularization term (default is 2).

        Returns:
            torch.Tensor: The computed modified loss value.
        """
        # Apply softmax to get probabilities
        gate_softmax = F.softmax(gate_logits, dim=-1)

        # Calculate entropy loss
        entropy_loss = -torch.sum(gate_softmax * torch.log(gate_softmax + 1e-12), dim=-1)

        # Calculate regularization term: sum of probabilities to the power of k
        regularization_term = torch.sum(gate_softmax ** k, dim=-1)

        # Calculate weighted entropy loss (1 / regularization_term)
        weighted_entropy_loss = (1 / (regularization_term + 1e-12)) * entropy_loss

        # Return the mean modified loss
        return weighted_entropy_loss.mean()

    def entropy_adv2(self, gate_logits, gate_softmax, k=2):
        """
        Computes the modified loss, which combines entropy loss with a regularization term.

        Args:
            gate_logits (torch.Tensor): The logits from the gating network.
            k (int): The power for the regularization term (default is 2).

        Returns:
            torch.Tensor: The computed modified loss value.
        """
        # Calculate entropy loss
        entropy_loss = -torch.sum(gate_softmax * torch.log(gate_softmax + 1e-12), dim=-1)

        # Calculate regularization term: sum of probabilities to the power of k
        regularization_term = torch.sum(gate_softmax ** k, dim=-1)

        # Calculate the maximum value in probabilities
        max_value = torch.max(gate_softmax, dim=-1).values

        # Calculate weighted entropy loss (1 - max_value) * (1 / regularization_term) * entropy_loss
        weighted_entropy_loss = (1 - max_value) * (1 / (regularization_term + 1e-12)) * entropy_loss

        # Return the mean of the weighted entropy loss
        return weighted_entropy_loss.mean()
    
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

    def compute_moe_dynamic(self, gate_logits, selected_experts, weights, results, x):
        with torch.no_grad():
            # selected_experts = selected_experts.squeeze(-1)
            # experts_union = build_expert_union(selected_experts, 
            #                     num_prev=self.number_of_previous_tokens)
            
            # experts_union = experts_union[:, :, :self.number_of_previous_tokens]
            x_padded = F.pad(selected_experts, pad=(0, 0, self.number_of_previous_tokens - 1, 0), mode='constant', value=-1)
            x_unfold = x_padded.unfold(dimension=1, size=self.number_of_previous_tokens, step=1)
            experts_union = x_unfold.squeeze(-1).squeeze(2)                        
            index_expert_unique = row_unique_and_sort(
                experts_union
            )
            mask = (index_expert_unique != -1)
            clipped_index = torch.clamp(index_expert_unique, min=0)

        
            gating_scores = gate_logits.gather(dim=2, index=clipped_index)
            # turn on it when need to gradient 
            # gating_scores = torch.where(mask, gating_scores, torch.full_like(gating_scores, float('-inf')))
            
            neg_inf = torch.tensor(float('-inf'), device=gating_scores.device, dtype=gating_scores.dtype)
            gating_scores.masked_fill_(~mask, neg_inf)
            
            
        gating_softmax = F.softmax(gating_scores, dim=-1)
        
        infor_experts = {}
        for i in range(self.num_of_experts):
            batch_idx, token_idx, topk_idx = torch.where(index_expert_unique== i)
            infor_experts[i] = [batch_idx, token_idx, topk_idx]
            
        for i, expert in enumerate(self.experts):
            batch_idx, token_idx, topk_idx = infor_experts[i]
            weights_slice = gating_softmax[batch_idx, token_idx, topk_idx].unsqueeze(0).T
            results[batch_idx, token_idx] +=weights_slice*expert(x[batch_idx, token_idx])
        return results
    
    def forward(self, x, return_id_experts = False, is_vision = False):
        # x_norm = torch.norm(x, dim=-1, keepdim=True)
        # if self.normalization :

        #     gate_logits = self.gate(self.layer_norm(x))``
        #     # x = x / x.norm(p=2, dim=-1, keepdim=True)
        #     # gate_logits = self.gate(x / x.norm(p=2, dim=-1, keepdim=True))
        # else:
        gate_logits = self.gate(x)

        weights, selected_experts = torch.topk(gate_logits, k=self.num_selected, dim=2)
        gate_softmax = F.softmax(gate_logits, dim=-1, dtype=torch.float).to(x.dtype)
        output = torch.zeros(x.shape[0], x.shape[1], self.out_embed_dim, device=x.device, dtype=x.dtype)

        auxiliary_loss = torch.tensor(0.0, device=x.device)

        balance_loss = torch.tensor(0.0, device=x.device)        
        # if self.dynamic_activate or x.requires_grad == False:
        output = self.compute_moe_dynamic(gate_logits = gate_softmax, selected_experts = selected_experts, weights = weights, results = output, x = x)
            
        
        if x.requires_grad == True:
            auxiliary_loss, balance_loss = self.combine_loss(selected_experts, gate_softmax, gate_logits)

        if return_id_experts:
            return output, auxiliary_loss, selected_experts
        else: 
            return output, auxiliary_loss, None, balance_loss
