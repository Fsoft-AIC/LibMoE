import torch
import torch.nn as nn
from einops import rearrange, repeat, reduce, pack, unpack
import torch.nn.functional as F
import copy



import torch
import torch.nn as nn
from einops import rearrange, repeat, reduce, pack, unpack
import torch.nn.functional as F
import copy


class MoeLayer(nn.Module):
    
    def __init__(self, in_embed_dim=768, out_embed_dim=768, num_of_experts=4, num_selected=2, expert=None, args=None):
        super().__init__()
        self.in_embed_dim = in_embed_dim
        self.out_embed_dim = out_embed_dim
        self.num_of_experts = num_of_experts
        self.num_selected = num_selected

        if expert is None:
            self.experts = nn.ModuleList([
                nn.Sequential(nn.Linear(in_embed_dim, out_embed_dim), 
                nn.GELU(), 
                nn.Linear(out_embed_dim, out_embed_dim)) for _ in range(num_of_experts)])
        else:
            if isinstance(expert, nn.ModuleList):
                print("Training from scratch ...")
                self.experts = expert
            else:
                self.experts = nn.ModuleList([copy.deepcopy(expert) for _ in range(self.num_of_experts)])
        
        self.gate = nn.Linear(in_embed_dim, num_of_experts, bias=False)
        self.args = args

    
    def zloss(self, gate_logits, gate_softmax = None):
        router_z_loss = torch.logsumexp(gate_logits, dim=-1)
        router_z_loss = torch.square(router_z_loss)            
        router_z_loss = router_z_loss.mean()
        return router_z_loss

    def balanceloss(self, selected_experts, gate_softmax):
        density_1_proxy = reduce(gate_softmax, '... n e -> ... e', 'mean')
        one_hot_gate_indices = nn.functional.one_hot(rearrange(selected_experts, '... k -> k ...'), self.num_of_experts).float()[0]
        density_1 = reduce(one_hot_gate_indices, '... n e -> ... e', 'mean')
        balance_loss = (density_1_proxy * density_1).mean() * float(self.num_of_experts ** 2)
        return balance_loss


    def topk_expert(self, gate_logits):
        gate_softmax = F.softmax(gate_logits, dim=-1, dtype=torch.float32)
        weights, selected_experts = torch.topk(gate_softmax, self.num_selected)
        
        return weights, selected_experts, gate_softmax
    
    def compute_moe(self, selected_experts, weights, results, x, expert_outputs = None):
        infor_experts = {}
        
        for i in range(len(self.experts)):
            batch_idx, token_idx, topk_idx = torch.where(selected_experts == i)
            infor_experts[i] = [batch_idx, token_idx, topk_idx]
        # if expert_outputs is not None:
        is_expert = expert_outputs is not None
        for i, expert in enumerate(self.experts):
            batch_idx, token_idx, topk_idx = infor_experts[i]
            if is_expert:
                out_exp = expert_outputs[i][batch_idx, token_idx]
            else:
                out_exp = expert(x[batch_idx, token_idx])
            results[batch_idx, token_idx] += weights[batch_idx, token_idx, topk_idx].unsqueeze(0).T * out_exp
        
        return results
    def combine_loss(self, selected_experts, gate_softmax, gate_logits):
        # compute balance loss
        balance_loss = self.balanceloss(selected_experts=selected_experts, gate_softmax=gate_softmax)
        # compute router_z_loss
        router_z_loss = self.zloss(gate_logits, gate_softmax)

        auxiliary_loss = balance_loss * self.args.balance_loss_coef + \
            router_z_loss * self.args.router_z_loss_coef
        
        return auxiliary_loss, balance_loss
    
    def forward(self, x, return_id_experts = False, is_vision = False):
        # compute output
        gate_logits = self.gate(x)
        weights, selected_experts, gate_softmax = self.topk_expert(gate_logits=gate_logits)
        weights = weights / torch.sum(weights, dim=-1, keepdim=True).to(x.dtype)
        output = torch.zeros(x.shape[0], x.shape[1], self.out_embed_dim, device=x.device, dtype=x.dtype)
        output = self.compute_moe(selected_experts, weights, output, x)
        # compute loss
        auxiliary_loss, balance_loss = self.combine_loss(selected_experts, gate_softmax, gate_logits)
        if return_id_experts:
            return output, auxiliary_loss, selected_experts, balance_loss
        else:
            return output, auxiliary_loss, None, balance_loss





class CopyExpert(torch.nn.Module):
    def __init__(self, expert):
        super(CopyExpert, self).__init__()

    def forward(self, inputs):
        return inputs


class ZeroExpert(torch.nn.Module):
    def __init__(self, expert):
        super(ZeroExpert, self).__init__()

    def forward(self, inputs):
        return torch.zeros_like(inputs).to(inputs.dtype).to(inputs.device)


class ConstantExpert(torch.nn.Module):
    def __init__(self, expert, out_embed_dim):
        super(ConstantExpert, self).__init__()
        self.constant = torch.nn.Parameter(
            torch.empty((out_embed_dim)))
        torch.nn.init.normal_(self.constant)

        self.wg = torch.nn.Linear(out_embed_dim, 2, bias=False)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, inputs):
        # print(inputs.size())
        weight = self.wg(inputs)
        weight = self.softmax(weight)
        return torch.einsum('b,bd->bd', [weight[:, 0].type_as(inputs), inputs]) + torch.einsum(
                'b,d->bd', [weight[:, 1].type_as(inputs), self.constant.type_as(inputs)])

class SMoEPlusPlus(MoeLayer):
    
    def __init__(self, in_embed_dim=768, out_embed_dim=768, num_of_experts=4, num_selected=2, expert=None, args=None):
        super().__init__(in_embed_dim, out_embed_dim, num_of_experts, num_selected, expert, args)
        
        self.num_constant_experts = 2
        self.use_moe_plus_plus = in_embed_dim == out_embed_dim
        
        if expert is None:
            # For MoE++
            if self.use_moe_plus_plus:
                self.zero_expert = ZeroExpert(expert)
                self.copy_expert = CopyExpert(expert)
                self.constant_experts = [ConstantExpert(expert) for _ in range(self.num_constant_experts)]
                
                self.experts = nn.ModuleList([
                    nn.Sequential(nn.Linear(in_embed_dim, out_embed_dim), nn.GELU(), nn.Linear(out_embed_dim, out_embed_dim)) for _ in range(self.num_of_experts)] + 
                    [self.zero_expert, self.copy_expert] + self.constant_experts)
            else:
                self.experts = nn.ModuleList([
                    nn.Sequential(nn.Linear(in_embed_dim, out_embed_dim), nn.GELU(), nn.Linear(out_embed_dim, out_embed_dim))
                    for _ in range(self.num_of_experts)])
            
        else:
            if isinstance(expert, nn.ModuleList):
                self.experts = expert
            else:
                if self.use_moe_plus_plus:
                    self.zero_expert = ZeroExpert(expert)
                    self.copy_expert = CopyExpert(expert)
                    self.constant_experts = [ConstantExpert(expert) for _ in range(self.num_constant_experts)]
                    self.experts = nn.ModuleList([copy.deepcopy(expert) for _ in range(self.num_of_experts)] + 
                                                [self.zero_expert, self.copy_expert] + self.constant_experts)
                else:
                    self.experts = nn.ModuleList([copy.deepcopy(expert) for _ in range(self.num_of_experts)])
        
        self.num_of_experts = self.num_of_experts + self.num_constant_experts + 2 if self.use_moe_plus_plus else self.num_of_experts
        
        self.gate = nn.Linear(in_embed_dim, self.num_of_experts, bias=False)
        self.args = args


    def forward(self, x, return_id_experts = False, is_vision = False):
        # compute output
        gate_logits = self.gate(x)
        weights, selected_experts, gate_softmax = self.topk_expert(gate_logits=gate_logits)
        weights = weights / torch.sum(weights, dim=-1, keepdim=True).to(x.dtype)
        output = torch.zeros(x.shape[0], x.shape[1], self.out_embed_dim, device=x.device, dtype=x.dtype)
        output = self.compute_moe(selected_experts, weights, output, x)
        # compute loss
        auxiliary_loss, balance_loss = self.combine_loss(selected_experts, gate_softmax, gate_logits)
        if return_id_experts:
            return output, auxiliary_loss, selected_experts, balance_loss
        else:
            return output, auxiliary_loss, None, balance_loss
