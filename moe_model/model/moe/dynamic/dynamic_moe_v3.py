
import torch
import torch.nn as nn
import torch.nn.functional as F
from moe_model.model.moe.register import register_moe
from moe_model.model.moe.moe import MoeLayer
from .utils import unique_each_row_vectorized, create_table_from_index_and_value
from datetime import datetime

@register_moe("dynamic_moe_v3")
class DynamicMoEv3(MoeLayer):
    
    def __init__(self, in_embed_dim=768, out_embed_dim=768, num_of_experts=4, num_selected=2, expert=None, args=None):
        super().__init__(in_embed_dim, out_embed_dim, num_of_experts, num_selected, expert, args)
        print("Starting with Dynamic MoE")
        self.gate = nn.Linear(in_embed_dim, num_of_experts, bias=True) 
        
        self.total_steps = None
        self.current_steps = 1
        self.num_current_selected = 1
        self.number_of_previous_tokens = args.number_of_previous_tokens
        self.normalization = args.normalization
        self.dynamic_activate = False
        self.num_selected = 1
        # Get the current time
        # Format the string with day, month, year, hour, minute, and second
        self.name_logs = datetime.now().strftime("`dynamic_`%Y-%m-%dT%H-%M-%S.%f.txt")
        assert self.number_of_previous_tokens <= self.num_of_experts, "The number of preceding tokens must be less than the number of experts. \n num_of_experts: {self.num_of_experts} \n number_of_previous_tokens: {number_of_previous_tokens}"
        self.aux_loss['eloss'] = self.entropy_loss
        self.aux_loss['eadv_loss'] = self.entropy_adv
        self.aux_loss['eadv2_loss'] = self.entropy_adv2
        print(f"Apply loss 1: {args.loss1}")
        print(f"Apply loss 2: {args.loss2}")
        self.loss1 = self.aux_loss[args.loss1]
        self.loss2 = self.aux_loss[args.loss2]
        self.strategy_train = args.strategy_train
        self.compute_loss_dynamic = self.combine_loss
        if self.strategy_train == "mix_loss":
            print(f"Apply Strategy training {self.strategy_train}")
            self.compute_loss_dynamic = self.strategy_mix_loss

        self.layer_norm = nn.Identity()
        self.last_percent = 0
        if self.args.normalization:
            print("Apply Normalization")
            self.layer_norm = nn.LayerNorm(in_embed_dim)

    def set_total_steps(self, step):
        self.total_steps = step

    def set_current_steps(self, step):
        self.current_steps = step
    # def strategy_learning(self, ) 
    def set_num_current_selected(self, is_log = False):
        # linear increase 
        # if self.current_steps > self.total_steps*0.2 and self.dynamic_activate == False:
        #     self.dynamic_activate = True
        #     print("Apply Dynamic Expert")
        #     with open("/cm/shared/anonymous_H102/toolkitmoe/log_dynamic.txt", 'a') as file:
        #         file.write("Apply Dynamic Expert\n")
        # print()
        # if self.total_steps == None:
        #     self.total_steps = 10
        percent_progress = int((self.current_steps / self.total_steps)*100)
        
        if percent_progress > 80:
            self.dynamic_activate = True
            return 
        if self.current_steps > 0 and percent_progress % 10 == 0 and percent_progress != self.last_percent:
            self.dynamic_activate = not self.dynamic_activate
            # Get the current time with microsecond precision
            current_time = datetime.now()
            # Format the timestamp in a professional log format (ISO 8601 with milliseconds)
            timestamp = current_time.strftime("%Y-%m-%dT%H-%M-%S.%f")[:-3]  # Trim microseconds to milliseconds
            if is_log:
                with open(f"/cm/shared/anonymous_H102/toolkitmoe/logs/{self.name_logs}", 'a') as file:
                    file.write(f"{timestamp}: Apply Dynamic Expert at {percent_progress}% data: {self.dynamic_activate}\n")

        self.last_percent = percent_progress
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


        # Calculate entropy
        entropy = -torch.sum(gate_softmax * torch.log(gate_softmax + 1e-12), dim=-1)

        # Return the mean entropy loss (lower entropy means less uniform distribution)
        return entropy.mean()
    
    

    
    def combine_loss(self, selected_experts, gate_softmax, gate_logits):
        
        # compute balance loss
        balance_loss = self.loss1(selected_experts=selected_experts, gate_softmax=gate_softmax)
        # compute router_z_loss
        entropy_loss = self.loss2(gate_logits, gate_softmax = gate_softmax)
        # print(f"Entropy: {entropy_loss}")
        # balance_loss = self.balanceloss(selected_experts=selected_experts, gate_softmax=gate_softmax)
        # entropy_loss = self.entropy_adv2(gate_logits = gate_logits, gate_softmax = gate_softmax)
        auxiliary_loss = balance_loss * self.args.balance_loss_coef + entropy_loss * self.args.router_z_loss_coef
        
        return auxiliary_loss
    def strategy_mix_loss(self, selected_experts, gate_softmax, gate_logits):
        if self.dynamic_activate == False:
            
            # compute balance loss
            loss = self.args.balance_loss_coef * self.balanceloss(selected_experts, gate_softmax) 
        else:

            # compute router_z_loss
            loss = self.args.balance_loss_coef * self.entropy_loss(gate_logits = gate_logits, gate_softmax = gate_softmax)
 
        return loss

    def compute_moe_dynamic(self, gate_logits, selected_experts, weights, results, x):
        # gate_logits = F.softmax(gate_logits, dim=-1)
        x_padded = F.pad(selected_experts, pad=(0, 0, self.number_of_previous_tokens-1, 0), mode='constant', value=-1) 
        
        x_unfold = x_padded.unfold(dimension=1, size=self.number_of_previous_tokens, step=1)  

        selected_experts_stride = x_unfold.squeeze(-1).squeeze(2)  # Kích thước (B, N, L)

        index_expert_unique = unique_each_row_vectorized(selected_experts_stride, -1)
        weight_after_unique, sum_count_row = create_table_from_index_and_value(index_expert_unique, value=gate_logits, fill_value=-float('inf'))
        # breakpoint()
        weight_softmax = F.softmax(weight_after_unique, dim=2)
        infor_experts = {}
        for i in range(self.num_of_experts):
            batch_idx, token_idx, topk_idx = torch.where(index_expert_unique== i)
            infor_experts[i] = [batch_idx, token_idx, topk_idx]
            
        for i, expert in enumerate(self.experts):
            batch_idx, token_idx, topk_idx = infor_experts[i]
            
            weights_slice = weight_softmax[batch_idx, token_idx, topk_idx].unsqueeze(0).T
            results[batch_idx, token_idx] +=weights_slice*expert(x[batch_idx, token_idx])
        # infor_experts = {}
        # for i in range(self.num_of_experts):
        #     batch_idx, token_idx, topk_idx = torch.where(index_expert_unique== i)
        #     infor_experts[i] = [batch_idx, token_idx, topk_idx]
            
        # for i, expert in enumerate(self.experts):
        #     batch_idx, token_idx, topk_idx = infor_experts[i]
        #     results[batch_idx, token_idx] +=expert(x[batch_idx, token_idx])
            
        # results = results / sum_count_row

        return results
    
    def forward(self, x, return_id_experts = False):
        if self.normalization :

            gate_logits = self.gate(self.layer_norm(x))
        else:
            gate_logits = self.gate(x)

        weights, selected_experts = torch.topk(gate_logits, k=self.num_selected, dim=2)
        gate_softmax = F.softmax(gate_logits, dim=-1, dtype=torch.float).to(x.dtype)
        output = torch.zeros(x.shape[0], x.shape[1], self.out_embed_dim, device=x.device, dtype=x.dtype)

        auxiliary_loss = torch.tensor(0.0, device=x.device)
        
        if x.requires_grad == True:

            
            if self.dynamic_activate :
                output = self.compute_moe_dynamic(gate_logits = gate_softmax, selected_experts = selected_experts, weights = weights, results = output, x = x)
            else:
                output = self.compute_moe(selected_experts, weights, output, x)
            auxiliary_loss = self.compute_loss_dynamic(selected_experts, gate_softmax, gate_logits)
            self.set_num_current_selected(x.device)
        else:
            
            output = self.compute_moe_dynamic(gate_logits = gate_softmax, selected_experts = selected_experts, weights = weights, results = output, x = x)

        if return_id_experts:
            return output, auxiliary_loss, selected_experts
        else: 
            return output, auxiliary_loss, None
