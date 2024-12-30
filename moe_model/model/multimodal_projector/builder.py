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


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}

class MLPMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_selected = config.num_selected
        self.mm_channels = (config.mm_hidden_size * len(config.scales))
        self.channels = config.hidden_size

        self.moelayer = get_moe(config.moe_name)(
            in_embed_dim = self.mm_channels, 
            out_embed_dim = self.channels, 
            num_of_experts = config.num_experts,
            num_selected = config.num_selected,
            expert = nn.Sequential(
                nn.Linear(self.mm_channels, self.channels), 
                nn.GELU(), 
                nn.Linear(self.channels, self.channels)
            ),
            args=config
        )
        
    def forward(self, x_img, return_id_experts = False):
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
        modules = [nn.Linear(config.mm_hidden_size * len(config.scales), config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)


    if projector_type == 'moe':
        return MLPMoE(config)
    else:
        if projector_type == 'identity':
            return IdentityMap()
        raise ValueError(f'Unknown projector type: {projector_type}')
