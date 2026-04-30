import torch
import torch.nn
from typing import Dict, Any


class LayerWithStats(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def get_stats(self) -> Dict[str, Any]:
        raise NotImplementedError()


class LayerStatProcessor:
    def __init__(self, module: torch.nn.Module):
        self.modules = []
        for n, m in module.named_modules():
            if isinstance(m, LayerWithStats):
                self.modules.append((n, m))

    def get(self) -> Dict[str, Any]:
        res = {}
        for n, m in self.modules:
            res.update({f"{n}/{k}": v for k, v in m.get_stats().items()})
        return res
