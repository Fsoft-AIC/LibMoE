
import torch
import torch.nn as nn
import torch.nn.functional as F
from .register import register_moe
from .moe import MoeLayer


@register_moe("hyperrouter")
class HyperRouter(MoeLayer):
    def __init__(self, in_embed_dim=768, out_embed_dim=768, num_of_experts=4, num_selected=2, expert=None, args=None):
        super().__init__(in_embed_dim, out_embed_dim, num_of_experts, num_selected, expert, args)

        # define HyperNetwork 
        hyper_size = in_embed_dim // 4
        hyper_embedding = torch.randn([1, in_embed_dim])
        self.register_parameter("hyper_embedding", nn.Parameter(hyper_embedding))

        self.hypernet = nn.Sequential(
            nn.Linear(in_embed_dim, hyper_size),
            nn.ReLU(),
            nn.Linear(hyper_size, in_embed_dim * num_of_experts + num_of_experts)
        ) 
        self.init_hypernet_weights()
        self.gate = nn.Linear(in_embed_dim, num_of_experts, bias=True) 
        
        # used for updating number of activated experts
        self.total_steps = None
        self.current_steps = 1
        self.num_current_selected = 1
        self.topk_max = args.topk_max 
        self.topk_min = args.topk_min
    def init_hypernet_weights(self):
        # Loop over all layers in self.hypernet and initialize weights and biases
        for layer in self.hypernet:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0.0, 0.02)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)

    def set_total_steps(self, step):
        self.total_steps = step

    def set_current_steps(self, step):
        self.current_steps = step
    
    def set_num_current_selected(self):
        number_experts = self.topk_max - self.topk_min
        # linear increase 
        self.num_current_selected = round(number_experts * self.current_steps / self.total_steps) + self.topk_min
    
    def forward(self, x,  return_id_experts = False,  is_vision = False):
        if x.requires_grad: # check if it is training phase
            self.set_num_current_selected()
            topk = self.num_current_selected 
            self.hypernet_outputs = self.hypernet(self.hyper_embedding)[0]
            weights_splice = self.hypernet_outputs.reshape([self.num_of_experts, -1]) # (num_of_experts, in_embed_dim + 1) 
            del self.gate.weight 
            self.gate.weight = weights_splice[:, :-1]
            del self.gate.bias 
            self.gate.bias = weights_splice[:, -1] 
        else: 
            topk = self.num_selected

        gate_logits = self.gate(x)

        weights, selected_experts = torch.topk(gate_logits, k=topk, dim=2)
                                            
        weights =  F.softmax(weights, dim=-1, dtype=torch.float).to(x.device)

        output = torch.zeros(x.shape[0], x.shape[1], self.out_embed_dim, device=x.device, dtype=x.dtype)
        output = self.compute_moe(selected_experts, weights, output, x)
        auxiliary_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        balance_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
   
        if return_id_experts:
            return output, auxiliary_loss, selected_experts, balance_loss
        else:
            return output, auxiliary_loss, None, balance_loss

