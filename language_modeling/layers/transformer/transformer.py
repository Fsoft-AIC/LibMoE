import torch
import math
from typing import Optional, Callable, Dict, Type, Sequence, Union

ActivationFunction = Callable[[torch.Tensor], torch.Tensor]

def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1)

def reset_prenorm_params(m: torch.nn.Module, n_layers: int):
    for layer in m.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.trunc_normal_(layer.weight)
            with torch.no_grad():
                layer.weight.mul_(math.sqrt(2 / (n_layers * layer.weight.shape[1])) / layer.weight.std())
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)
        elif isinstance(layer, torch.nn.LayerNorm):
            torch.nn.init.ones_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
