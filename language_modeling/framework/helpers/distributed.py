import os
import torch.distributed
from typing import Tuple
import datetime
# import deepspeed
from typing import Optional, Any

hostlist = None


class DDPWrapper:
    def __init__(self, module: torch.nn.Module) -> None:
        self._module = module
        self._ddp_module = torch.nn.parallel.DistributedDataParallel(self._module, broadcast_buffers=False, find_unused_parameters=True)

    def parameters(self, *args, **kwargs):
        return self._ddp_module.parameters(*args, **kwargs)

    def named_parameters(self, *args, **kwargs):
        return self._ddp_module.named_parameters(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self._ddp_module(*args, **kwargs)

    def set_grad_sync(self, enabled: bool):
        self._ddp_module.require_backward_grad_sync = enabled

    def __call__(self, *args, **kwds):
        return self._ddp_module(*args, **kwds)

    def __getattr__(self, item):
        return getattr(self._module, item)

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name in {"_module", "_ddp_module"}:
            super().__setattr__(__name, __value)
        else:
            setattr(self._module, __name, __value)


def has_extra_work(len: int, world_size: Optional[int], rank: Optional[int]) -> bool:
    if (world_size or 1) == 1:
        return False

    rem = len % world_size
    return rank < rem


def is_work_uneven(len: int, world_size: Optional[int]) -> bool:
    return world_size is not None and len % world_size != 0


def get_work_slice(len: int, world_size: Optional[int], rank: Optional[int]) -> Tuple[int, int]:
    if (world_size or 1) == 1:
        return 0, len

    assert rank is not None, "If world_size > 1, rank must be specified"
    rem = len % world_size

    real_batch_size = len // world_size
    batch_offset = real_batch_size * rank + min(rank, rem)
    real_batch_size += int(rank < rem)

    return batch_offset, real_batch_size


class DistributedEnv:
    is_distributed: bool
    rank: int
    world_size: int

    def __init__(self) -> None:
        self.is_distributed = False

        self.detect_env()

        if not self.is_distributed:
            self.world_size = 1
            self.rank = 0

    def detect_env(self):
        pass

    def init_env(self):
        pass

    def is_preemtible(self):
        return False

    def is_restart(self):
        return False

    def get_run_identifier(self):
        raise NotImplementedError()

    def is_master(self):
        return (not self.is_distributed) or (self.rank == 0)

    def has_extra_work(self, work_size: int) -> bool:
        return self.is_distributed and has_extra_work(work_size, self.world_size, self.rank)

    def is_work_uneven(self, work_size: int) -> bool:
        return self.is_distributed and is_work_uneven(work_size, self.world_size)

    def __repr__(self) -> str:
        if not self.is_distributed:
            return "No distributed environment detected"
        else:
            return f"Distributed environment: {self.__class__.__name__}. World size: {self.world_size}, rank: {self.rank}"


class SLURMEnv(DistributedEnv):
    def detect_env(self):
        global hostlist

        self.rank = os.getenv("SLURM_PROCID")
        self.world_size = os.getenv("SLURM_NPROCS")
        self.hostnames = os.getenv('SLURM_JOB_NODELIST')
        self.gpu_ids = os.getenv('SLURM_STEP_GPUS')
        self.local_id = os.getenv('SLURM_LOCALID')
        self.restart_count = os.getenv('SLURM_RESTART_COUNT')
        self.jobid = os.getenv('SLURM_JOB_ID')
        self.taskid = os.getenv('SLURM_ARRAY_TASK_ID')

        self.restart_count = int(self.restart_count) if self.restart_count is not None else 0

        self.is_distributed = self.rank is not None and self.world_size is not None and self.hostnames is not None and\
                              self.gpu_ids is not None and self.local_id is not None

        if self.is_distributed:
            if hostlist is None:
                import hostlist

            self.rank = int(self.rank)
            self.world_size = int(self.world_size)
            self.hostnames = hostlist.expand_hostlist(self.hostnames)
            self.gpu_ids = self.gpu_ids.split(",")
            self.local_id = int(self.local_id)

            self.port = 12345 + int(min(self.gpu_ids))

            self.is_distributed = self.world_size > 1

            print(f"SLURM env: rank {self.rank}, world size {self.world_size}, restart count {self.restart_count}")

            if self.local_id >= len(self.gpu_ids):
                raise ValueError(f"More tasks on a sigle node than GPUs ({self.local_id} >= {self.gpu_ids})")

    def init_env(self):
        if not self.is_distributed:
            return

        print(f"Initializing distributed environment. World size: {self.world_size}, master: {self.hostnames[0]}, "
              f"my rank {self.rank}")

        # SLURM hides the other GPUs, so set it just to the local id
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.local_id)
        torch.distributed.init_process_group('nccl', rank=self.rank, world_size=self.world_size,
                                             init_method=f"tcp://{self.hostnames[0]}:{self.port}",
                                             timeout=datetime.timedelta(0, 6000))
        # deepspeed.init_distributed()

    def is_preemtible(self):
        return True

    def is_restart(self):
        return self.restart_count > 0

    def get_run_identifier(self):
        return f"slurm_{self.jobid}_{self.taskid}"

    def __repr__(self) -> str:
        r = super().__repr__()
        if self.is_distributed:
            r = r + f" hostnames: {self.hostnames}, gpu_ids: {self.gpu_ids}"
        return r


class LocalEnv(DistributedEnv):
    def detect_env(self):
        self.world_size = os.getenv('WORLD_SIZE')
        self.rank = os.getenv('RANK')
        self.local_rank = os.getenv('LOCAL_RANK')
        self.master_addr = os.getenv('MASTER_ADDR')
        self.master_port = os.getenv('MASTER_PORT')

        self.is_distributed = self.rank is not None and self.world_size is not None and \
                              self.master_addr is not None and self.master_port is not None \
                              and self.local_rank is not None

        if self.is_distributed:
            self.rank = int(self.rank)
            self.local_rank = int(self.local_rank)
            self.world_size = int(self.world_size)

    def init_env(self):
        if not self.is_distributed:
            return

        torch.cuda.set_device(self.local_rank)

        print(f"Initializing local multigpu environment. World size: {self.world_size}, my rank {self.rank}")
        torch.distributed.init_process_group('nccl', rank=self.rank, world_size=self.world_size,
                                             init_method=f"tcp://{self.master_addr}:{self.master_port}")
        # deepspeed.init_distributed()
        print("Done initializing local multigpu environment")

    def __repr__(self) -> str:
        r = super().__repr__()
        if self.is_distributed:
            r = r + f" local_rank: {self.local_rank}, master_addr: {self.master_addr}, master_port: {self.master_port}"
        return r


def get_dist_env() -> DistributedEnv:
    envs = [SLURMEnv, LocalEnv]

    for env in envs:
        e = env()
        if e.is_distributed:
            return e

    return DistributedEnv()
