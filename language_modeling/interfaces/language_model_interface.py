import torch
import torch.nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from .model_interface import ModelInterface
from framework.interfaces import RecurrentResult
from framework.utils import U
import framework
import random
from framework.helpers.distributed import DistributedEnv


class LanguageModelResult(RecurrentResult):
    @property
    def batch_size(self) -> int:
        o = self.outputs.output if isinstance(self.outputs, torch.nn.modules.adaptive._ASMoutput) else self.outputs
        return o.shape[self.batch_dim]


class LanguageModelInterface(ModelInterface):
    def __init__(self, model: torch.nn.Module, batch_dim: int = 1, drop_state_prob: float = 0,
                 dist_env: Optional[DistributedEnv] = None, save_state: bool = False,
                 n_ubatches: Optional[int] = 1,
                 log: bool = False, mask_name: Optional[str] = "mask"):
        super().__init__()
        self.model = model
        self.batch_dim = batch_dim
        self.drop_state_prob = drop_state_prob
        self.time_dim = 1 - self.batch_dim
        self.dist_env = dist_env
        self.save_state = save_state
        self.n_ubatches = n_ubatches or 1
        self.log = log
        self.mask_name = mask_name
        self.reset_state()

    def create_input(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        return data["data"].narrow(self.time_dim, 0, data["data"].shape[self.time_dim] - 1)

    def decode_outputs(self, outputs: RecurrentResult) -> Any:
        return outputs.outputs

    def reset_state(self):
        self.state = [None] * self.n_ubatches

    def loss(self, net_out: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor], log: bool) -> Tuple[torch.Tensor, Dict[str, Any]]:
        assert net_out.shape[:-1] == target.shape
        assert net_out.ndim == 3

        target = target.flatten().long()
        if mask is not None:
            target = target.masked_fill(~(mask.flatten().bool()), -100)

        if log:
            l = F.cross_entropy(net_out.flatten(0, -2), target.flatten().long(), reduction='none', ignore_index=-100).view_as(target)
            loss = l.mean()
            loss_per_pos = l.detach().mean(1)
            loss_per_pos_l = loss_per_pos.cpu().numpy().tolist()
            csum = loss_per_pos.cumsum(0)
            csum = csum[-1] + loss_per_pos[-1] - csum
            nelem = torch.arange(loss_per_pos.shape[0], 0, -1, device=loss_per_pos.device)
            mean_until_end = csum / nelem
            mean_until_end = mean_until_end.cpu().numpy().tolist()
            return loss, {
                "loss_per_pos": framework.visualize.plot.XYChart({"Loss per pos": list(zip(range(len(loss_per_pos_l)), loss_per_pos_l))}),
                "loss_from_pos": framework.visualize.plot.XYChart({"Loss from this pos": list(zip(range(len(mean_until_end)), mean_until_end))}),
            }
        else:
            return F.cross_entropy(net_out.flatten(0, -2), target, ignore_index=-100), {}

    def create_target(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        return data["data"].narrow(self.time_dim, 1, data["data"].shape[self.time_dim] - 1).contiguous()

    def __call__(self, data: Dict[str, torch.Tensor], iter: int, ubatch: int) -> Tuple[LanguageModelResult, Dict[str, Any]]:
        if self.model.training and self.drop_state_prob > 0 and random.random() < self.drop_state_prob:
            self.reset_state()

        if ubatch > 0 and not self.model.training:
            raise ValueError("Microbatching is not supported in eval time")

        input = self.create_input(data)
        target = self.create_target(data)

        mask = data[self.mask_name].narrow(self.time_dim, 1, data[self.mask_name].shape[self.time_dim] - 1).contiguous() if (self.mask_name is not None and self.mask_name in data) else None

        plots = {}
        res, state = self.model(input, target, self.state[ubatch])
        if isinstance(res, torch.nn.modules.adaptive._ASMoutput):
            loss = res.loss
            # res = res.outputs
        else:
            loss, plots = self.loss(res, target, mask, self.log and (iter % 100 == 0))

        self.state[ubatch] = U.apply_to_tensors(state, lambda x: x.detach())
        return LanguageModelResult(res, loss), plots

    def state_dict(self) -> Dict[str, Any]:
        if not self.save_state:
            return {}

        if self.dist_env is not None and self.dist_env.is_distributed:
            # Collect the state from all workers
            state = []
            for s in self.state:
                alist = [None] * self.dist_env.world_size
                s = torch.distributed.all_gather(alist, s)
                s = torch.cat(s, self.batch_dim)
                state.append(s)
            return {"state": state}
        else:
            return {"state": self.state}

    def load_state_dict(self, state: Dict[str, Any]):
        if not self.save_state:
            self.reset_state()
            return

        if len(self.state) != len(state["state"]):
            print(f"WARNING: Number of microbatches changed from {len(state['state'])} to {len(self.state)}. Resetting state.")
            self.reset_state()
            return

        if self.dist_env is not None and self.dist_env.is_distributed:
            state_bs = state["state"][0].shape[self.batch_dim]
            if state_bs % self.dist_env.world_size != 0:
                print(f"WARNING: State batch size ({state_bs}) is not divisible by the number of workers ({self.dist_env.world_size}). Resetting state.")
                self.reset_state()
            else:
                bs_per_worker = state_bs // self.dist_env.world_size
                self.state = [s.narrow(self.batch_dim, self.dist_env.local_rank * bs_per_worker, bs_per_worker) for s in state["state"]]
        else:
            self.state = state["state"]
