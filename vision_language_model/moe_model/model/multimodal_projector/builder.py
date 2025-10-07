#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# ------------------------------------------------------------------------
# Modified from LLaVA (https://github.com/haotian-liu/LLaVA)
# Copyright 2024 Jiachen Li
# ------------------------------------------------------------------------

import torch.nn as nn
import re
import torch.nn.functional as F

from moe_model.model.moe.register import get_moe
import torch
from collections import OrderedDict

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}
class ExpertMLP(nn.Module):
    def __init__(self, mm_channels, channels):
        super().__init__()
        self.activation_fn = nn.GELU()
        self.fc1 = nn.Linear(mm_channels, channels)
        self.fc2 = nn.Linear(channels, channels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states
    
class PixelShuffle(nn.Module):
    def __init__(self, scale_factor=1):
        super().__init__()
        self.scale_factor = scale_factor
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale_factor == 1: return x
        bsz, seq, embed_dim = x.size()
        seq_root = int(seq**0.5)
        assert seq_root**2 == seq # Sequence length must be a perfect square for pixel shuffle
        assert seq_root % self.scale_factor == 0 # Sequence root must be divisible by scale factor

        height = width = seq_root
        x = x.view(bsz, height, width, embed_dim)
        h_out = height // self.scale_factor
        w_out = width // self.scale_factor
        
        x = x.reshape(bsz, h_out, self.scale_factor, w_out, self.scale_factor, embed_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(bsz, h_out * w_out, embed_dim * self.scale_factor**2)
        
        return x
class MLPMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_selected = config.num_selected
        self.mm_channels = (config.mm_hidden_size * len(config.scales))
        self.channels = config.hidden_size
        self.sparse_upcycling = config.sparse_upcycling
        if config.sparse_upcycling:
            expert = nn.Sequential(
                    nn.Linear(self.mm_channels, self.channels), 
                    nn.GELU(), 
                    nn.Linear(self.channels, self.channels)
                )
        else:
            print("Training from scratch")
            self.layer_norm = nn.LayerNorm(self.mm_channels, eps=1e-06)
            expert = nn.ModuleList([nn.Sequential(
                    nn.Linear(self.mm_channels, self.channels), 
                    nn.GELU(), 
                    nn.Linear(self.channels, self.channels)) for _ in range(config.num_experts)])
        
        self.moelayer = get_moe(config.moe_name)(
            in_embed_dim = self.mm_channels, 
            out_embed_dim = self.channels, 
            num_of_experts = config.num_experts,
            num_selected = config.num_selected,
            expert = expert,
            args=config
        )
        
        self.is_norm = getattr(self.config, "is_norm", False)
        if self.is_norm:
            self.layer_norm = nn.LayerNorm(self.mm_channels, eps=1e-06)
            
      
    def forward(self, x_img, return_id_experts = False):
        
        # with moe layer pretrain from pft stage, we just training MoE layer and part rest freeze parameter
        if self.moelayer.experts[0][0].weight.requires_grad:
            x_img = x_img.requires_grad_(True)
        
        results, auxiliary_loss, mlp_id_experts, balance_losses = self.moelayer(x_img, return_id_experts)
        return results, auxiliary_loss, mlp_id_experts, balance_losses

    @property
    def config(self):
        return {"mm_projector_type": 'smoe_mlp'}

def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:        
        mlp_depth = int(mlp_gelu_match.group(1))

        modules = OrderedDict([
            ("0", nn.Linear(config.mm_hidden_size * len(config.scales),
                            config.hidden_size)),
        ])
        idx = 1  
        for i in range(1, mlp_depth):    
            modules[str(idx)] = nn.GELU()  
            modules[str(idx+1)]   = nn.Linear(config.hidden_size, config.hidden_size)      
            idx += 2             
        projector = nn.Sequential(modules)
        return projector
    if projector_type == 'moe':
        return MLPMoE(config)
    else:
        if projector_type == 'identity':
            return IdentityMap()
        raise ValueError(f'Unknown projector type: {projector_type}')
