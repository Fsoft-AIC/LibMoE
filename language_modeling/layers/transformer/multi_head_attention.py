import torch
import torch.nn
from typing import Optional
from dataclasses import dataclass


@dataclass
class AttentionMask:
    src_length_mask: Optional[torch.Tensor]
    position_mask: Optional[torch.Tensor]
