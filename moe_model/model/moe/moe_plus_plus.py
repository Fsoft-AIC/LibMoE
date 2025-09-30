import torch
import torch.nn as nn
from einops import rearrange, repeat, reduce, pack, unpack
import torch.nn.functional as F
import copy

from .register import register_moe
from .moe import MoeLayer


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
    def __init__(self, expert, out_embed_dim, init_weight = True):
        super(ConstantExpert, self).__init__()
        self.constant = torch.nn.Parameter(
            torch.empty((out_embed_dim)))
        torch.nn.init.normal_(self.constant, mean=0.0, std=0.002)
        # self.constant = copy.deepcopy(expert.fc2.bias)
        # torch.nn.init.normal_(self.constant, mean=0.0, std=0.0002)
        self.wg = torch.nn.Linear(out_embed_dim, 2, bias=False)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.init_weight = init_weight
        # breakpoint()

        self.init_gate_weights()
    
    def init_gate_weights(self, std = 0.02):
        """
            Initialize the weights and bias of the gating layer.
            We are make sure that gating of the xmoe same init weight setting with other algorithms 
        """
        
        if self.init_weight == False:
            print("Not init weight")
            return 
        device = self.wg.weight.device if self.wg.weight.device != torch.device('meta') else torch.device('cpu')
        # device = self.wg.weight.device
        gate_generator = torch.Generator(device=device)
        gate_generator.manual_seed(42)
        """
        Initialize the weights and bias of the gating layer.
        """
        nn.init.normal_(self.wg.weight, mean=0.0, std=std, generator=gate_generator)
        if self.wg.bias is not None:
            nn.init.constant_(self.wg.bias, 0.0)
        print(f"Initializing weights and bias of the Constant Experts succefull with device: {device}")
    def forward(self, inputs):
        # print(inputs.size())
        weight = self.wg(inputs)
        # weight = self.softmax(weight)
        weight = F.softmax(weight, dim=-1, dtype=torch.float32)
        return torch.einsum('b,bd->bd', [weight[:, 0].type_as(inputs), inputs]) + torch.einsum(
                'b,d->bd', [weight[:, 1].type_as(inputs), self.constant.type_as(inputs)])

@register_moe("smoe_plus_plus")
class SMoEPlusPlus(MoeLayer):
    
    def __init__(self, in_embed_dim=768, out_embed_dim=768, num_of_experts=4, num_selected=2, expert=None, args=None):
        super().__init__(in_embed_dim, out_embed_dim, num_of_experts, num_selected, expert, args)
        
        self.num_constant_experts = 1
        self.use_moe_plus_plus = in_embed_dim == out_embed_dim
        self.out_embed_dim = out_embed_dim
        self.save_gate = None
        
        if expert is None:
            # # For MoE++
            # if self.use_moe_plus_plus:
            #     self.zero_expert = ZeroExpert(expert)
            #     self.copy_expert = CopyExpert(expert)
            #     self.constant_experts = [ConstantExpert(expert) for _ in range(self.num_constant_experts)]
                
            #     self.experts = nn.ModuleList([
            #         nn.Sequential(nn.Linear(in_embed_dim, out_embed_dim), nn.GELU(), nn.Linear(out_embed_dim, out_embed_dim)) for _ in range(self.num_of_experts)] + 
            #         [self.zero_expert, self.copy_expert] + self.constant_experts)
            # else:
            self.experts = nn.ModuleList([
                    nn.Sequential(nn.Linear(in_embed_dim, out_embed_dim), nn.GELU(), nn.Linear(out_embed_dim, out_embed_dim))
                    for _ in range(self.num_of_experts)])
            
        else:
            if isinstance(expert, nn.ModuleList):
                self.experts = expert
            else:
                # if self.use_moe_plus_plus:
                #     self.zero_expert = ZeroExpert(expert)
                #     self.copy_expert = CopyExpert(expert)
                #     self.constant_experts = [ConstantExpert(expert, out_embed_dim) for _ in range(self.num_constant_experts)]
                #     self.experts = nn.ModuleList([copy.deepcopy(expert) for _ in range(self.num_of_experts)] + 
                #                                 [self.zero_expert, self.copy_expert] + self.constant_experts)                    
                # else:
                self.experts = nn.ModuleList([copy.deepcopy(expert) for _ in range(self.num_of_experts)])
        
        self.num_of_experts = self.num_of_experts + self.num_constant_experts + 1 if self.use_moe_plus_plus else self.num_of_experts
        
        self.gate = nn.Linear(in_embed_dim, self.num_of_experts, bias=False)
        if self.use_moe_plus_plus:
            self.gate_transform = nn.Linear(self.num_of_experts, self.num_of_experts, bias=False)
        self.args = args
        self.std_gate = getattr(self.args, "std_gate", 0.02)
        self.init_gate_weights(self.std_gate)
        self.init_moe_plus()
    def init_gate_weights(self, std = 0.02):
        """
            Initialize the weights and bias of the gating layer.
            We are make sure that gating of the xmoe same init weight setting with other algorithms 
        """
        init_weight = getattr(self.args, "init_weight", True)

        if init_weight == False:
            print("Not init weight")
            return 
        device = self.gate.weight.device if self.gate.weight.device != torch.device('meta') else torch.device('cpu')
        # device = self.gate.weight.device
        gate_generator = torch.Generator(device=device)
        gate_generator.manual_seed(42)
        """
        Initialize the weights and bias of the gating layer.
        """
        nn.init.normal_(self.gate.weight, mean=0.0, std=std, generator=gate_generator)
        if self.use_moe_plus_plus:
            nn.init.normal_(self.gate_transform.weight, mean=0.0, std=std, generator=gate_generator)
        
        if self.gate.bias is not None:
            nn.init.constant_(self.gate.bias, 0.0)
        print(f"Initializing weights and bias of the gating layer succefull with device: {device}")
    def init_moe_plus(self, ):
        if self.use_moe_plus_plus:
          
            self.zero_expert = ZeroExpert(copy.deepcopy(self.experts[0]))
            # self.copy_expert = CopyExpert(copy.deepcopy(self.experts[0]))
            self.constant_experts = [ConstantExpert(self.experts[0], self.out_embed_dim) for _ in range(self.num_constant_experts)]
            
            self.experts = self.experts + [self.zero_expert] + self.constant_experts
            
            print("MoE++ setting successfull !")
    def balanceloss(self, selected_experts, gate_softmax):
        """
        Computes the balance loss for the selected experts.

        This loss measures how evenly the gating softmax distribution is distributed
        among the selected experts. It encourages a balanced distribution across experts
        by comparing the density of selected experts with the density of the overall gating softmax.

        Args:
            selected_experts (torch.Tensor): Indices of the selected experts 
            gate_softmax (torch.Tensor): Softmax probabilities for each expert 

        Returns:
            torch.Tensor: The computed balance loss value.
        """        
        density_1_proxy = reduce(gate_softmax, '... n e -> ... e', 'mean')
        one_hot_gate_indices = nn.functional.one_hot(rearrange(selected_experts, '... k -> k ...'), self.num_of_experts).float()[0]
        density_1 = reduce(one_hot_gate_indices, '... n e -> ... e', 'mean')
        dens = (density_1_proxy * density_1)
        # add t = 0.75 for avoid gate priority select zero 
        t = torch.ones_like(dens)
        if self.use_moe_plus_plus:
            # beacause we use less expert so need focus 
            t[:, 4] = 1.3

        balance_loss = (t*dens).mean() * float(self.num_of_experts ** 2)
        
        return balance_loss
    def forward(self, x, return_id_experts = False, is_vision = False, out_gate_prev = None):
        gate_logits = self.gate(x)
        # compute output
        # if  out_gate_prev is not None:
        #     gate_logits += self.gate_transform(out_gate_prev)
        # self.save_gate = gate_logits
        
        weights, selected_experts, gate_softmax = self.topk_expert(gate_logits=gate_logits)
        weights = weights / torch.sum(weights, dim=-1, keepdim=True).to(x.dtype)
        output = torch.zeros(x.shape[0], x.shape[1], self.out_embed_dim, device=x.device, dtype=x.dtype)
        output = self.compute_moe(selected_experts, weights, output, x, return_topk_outputs=False)
        auxiliary_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        balance_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        router_z_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        infor_aux = {}
        if self.gate.weight.requires_grad:
            # compute loss
            auxiliary_loss, balance_loss, router_z_loss = self.combine_loss(selected_experts, gate_softmax, gate_logits)
            infor_aux = {
                    "balance_loss": balance_loss.clone().detach(),
                    "router_z_loss": router_z_loss.clone().detach()
                }
        if self.gate.weight.requires_grad == False and return_id_experts == True:
            self.log_metrics['weights'] = weights
            self.log_metrics['balance_loss'] = balance_loss.item()
            self.log_metrics['router_z_loss'] = router_z_loss.item()
            self.log_metrics['gate_softmax'] = gate_softmax
            self.log_metrics['selected_experts'] = selected_experts
            self.log_metrics['router_magine'] = weights[:, :, 0] - weights[:, :, 1]

        return output, auxiliary_loss, None, infor_aux