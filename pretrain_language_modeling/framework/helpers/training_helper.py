import os
import shutil
import socket
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from typing import Any, Callable, List, Optional, Union, Dict

import numpy as np
import torch
import torch.distributed

from ..data_structures import DotDict
from ..utils import U, seed, use_gpu
from ..visualize import plot
from .. import visualize
from .argument_parser import ArgumentParser
from .distributed import get_dist_env, get_work_slice
from .saver import Saver, PyObjectSaver
import re


def get_plot_config(args):
    assert args.log in ["all", "tb", "wandb"]
    return args.log in ["all", "tb"], args.log in ["all", "wandb"]


def master(func):
    def wrapper(self, *args, **kwargs):
        if self.dist_env.is_master():
            func(self, *args, **kwargs)

    return wrapper


class TrainingHelper:
    args: DotDict

    find_slash = re.compile(r'/+')
    remove_firstlast_slash = re.compile(r'^/|/$')

    class Dirs:
        pass

    def __init__(self, register_args: Optional[Callable[[ArgumentParser], None]],
                 wandb_project_name: Optional[str] = None,
                 log_async: bool = False, extra_dirs: List[str] = [], restore: Optional[str] = None):

        self.dist_env = get_dist_env()
        self.dist_env.init_env()

        self.is_sweep = False
        self.log_async = log_async
        self.wandb_project_name = wandb_project_name
        self.all_dirs = ["checkpoint", "tensorboard"] + extra_dirs
        self.create_parser()
        self.last_saved = -1

        if register_args is not None:
            register_args(self.arg_parser)
        self.start(restore)

    def print_env_info(self):
        try:
            import pkg_resources
            print("---------------- Environment information: ----------------")
            installed_packages = pkg_resources.working_set
            print(list(sorted(["%s==%s" % (i.key, i.version) for i in installed_packages])))
            print("----------------------------------------------------------")
        except:  # noqa: E722
            pass

        try:
            git = subprocess.run(["git", "rev-parse", "--verify", "HEAD"], stderr=subprocess.DEVNULL,
                                 stdout=subprocess.PIPE)

            if git.returncode == 0:
                print(f"Git hash: {git.stdout.decode().strip()}")
        except:  # noqa: E722
            pass

    def create_parser(self):
        self.arg_parser = ArgumentParser(get_train_dir=lambda x: os.path.join("save", x.name) if x.name is not None
                                         else None)
        self.arg_parser.add_argument("-name", type=str, help="Train dir name",
                                     default=lambda x: os.path.join("save", x.name) if x.name is not None else None)
        self.arg_parser.add_argument("-reset", default=False, help="reset training - ignore saves", save=False)
        self.arg_parser.add_argument("-log", default="tb")
        self.arg_parser.add_argument("-save_interval", default="5000", parser=self.arg_parser.int_or_none_parser)
        self.arg_parser.add_argument("-wandb_save_interval", default="None", parser=self.arg_parser.int_or_none_parser)
        self.arg_parser.add_argument("-seed", default="none", parser=self.arg_parser.int_or_none_parser)
        self.arg_parser.add_argument("-gpu", default="auto", help="use this gpu")
        self.arg_parser.add_argument("-keep_alive", default=False)
        self.arg_parser.add_argument("-sweep_id_for_grid_search", default=0,
                                     help="Doesn't do anything, just to run multiple W&B iterations.")
        self.arg_parser.add_argument("-restore", default="")
        self.arg_parser.add_argument("-wandb_bug_workaround", default=False)
        self.arg_parser.add_argument("-wandb_sync_checkpoints", default=False)

    @master
    def create_dirs(self):
        self.dirs = self.Dirs()
        self.dirs.base = self.summary.save_dir

        for d in self.all_dirs:
            assert d not in self.dirs.__dict__, f"Directory {d} already exists"
            self.dirs.__dict__[d] = os.path.join(self.dirs.base, d)

        if self.use_wandb and not self.args.wandb_sync_checkpoints:
            ckpt_full = os.path.realpath(self.dirs.__dict__["checkpoint"])
            my_dir = os.path.realpath(os.getcwd())
            self.dirs.__dict__["checkpoint"] = f"save/{os.path.relpath(ckpt_full, my_dir)}"
            print(f"W&B: not synching checkpoints. Checkpoint save dir set to: {self.dirs.__dict__['checkpoint']}")

        if self.args.reset:
            print("Resetting training state...")
            for d in self.all_dirs:
                shutil.rmtree(self.dirs.__dict__[d], ignore_errors=True)

        for d in self.all_dirs:
            os.makedirs(self.dirs.__dict__[d], exist_ok=True)

    @master
    def save_startup_log(self):
        self.arg_parser.save(os.path.join(self.summary.save_dir, "args.json"))
        with open(os.path.join(self.summary.save_dir, "startup_log.txt"), "a+") as f:
            f.write(f"{str(datetime.now())} {socket.gethostname()}: {' '.join(sys.argv)}\n")

    @master
    def start_tensorboard(self):
        if self.use_tensorboard:
            os.makedirs(self.dirs.tensorboard, exist_ok=True)
            visualize.tensorboard.start(log_dir=self.dirs.tensorboard)

    def use_cuda(self) -> bool:
        return torch.cuda.is_available() and self.args.gpu.lower() != "none"

    def setup_environment(self):
        use_gpu(self.args.gpu)
        if self.args.seed is not None:
            assert not self.dist_env.is_distributed
            seed.fix(self.args.seed)

        self.device = torch.device(f"cuda:{torch.cuda.current_device()}") if self.use_cuda() else torch.device("cpu")

    def get_batch_size(self, full_batch_size: Optional[int] = None) -> int:
        batch_size = full_batch_size or self.args.batch_size
        if self.dist_env.is_distributed:
            return get_work_slice(batch_size, self.dist_env.world_size, self.dist_env.rank)[1]
        else:
            return batch_size

    def get_loss_scaling(self) -> float:
        # Scale that accounts for uneven world sizes. For mean reduction
        return self.get_batch_size() / self.args.batch_size

    def get_job_record_name(self) -> str:
        return f"jobs/{self.dist_env.get_run_identifier()}"

    def handle_env_restart(self) -> Optional[str]:
        if self.dist_env.is_restart():
            print("Restart detected. Restoring training state...")
            jobinfo = self.get_job_record_name()
            if not os.path.exists(jobinfo):
                raise ValueError(f"Restarting, but job record not found for jobid '{jobinfo}'. Exiting.")

            with open(jobinfo, "r") as f:
                checkpoint_dir = f.read().strip()

            return checkpoint_dir
        else:
            return None

    @master
    def record_job_info(self):
        if self.dist_env.is_preemtible():
            record = self.get_job_record_name()
            os.makedirs(os.path.dirname(record), exist_ok=True)
            with open(record, "w") as f:
                f.write(self.dirs.checkpoint)

    def find_checkpoint(self, name: str) -> str:
        if not os.path.isdir(name):
            return name

        # If we got a directory, assume it has the checkpoint, and it is model-<index>.pth, and find the last one.
        checkpoints = [int(f.split("-")[1].split(".")[0]) for f in os.listdir(name) if f.startswith("model-") and f.endswith(".pth")]

        if not checkpoints:
            raise ValueError(f"Resume: directory given ({name}), but no checkpoints found.")

        return os.path.join(name, f"model-{max(checkpoints)}.pth")

    def start(self, restore: Optional[str]):
        self.args = self.arg_parser.parse_and_try_load()
        self.restore_pending = None
        self.wandb_bug_found = False

        if self.dist_env.is_master():
            restore_env = self.handle_env_restart()
            if restore_env:
                restore = restore_env

            if restore or self.args.restore:
                # Restore args first such that the rest of the config is loaded correctly. Do not restore the GPU settings.
                restore_name = restore or self.args.restore
                restore_name = self.find_checkpoint(restore_name)

                print(f"Restoring: {restore_name}...")

                gpu_backup = self.args.gpu
                reset_backup = self.args.reset
                self.restore_pending = Saver.do_load(restore_name)
                self.args = self.arg_parser.from_dict(self.restore_pending["run_invariants"]["args"])
                self.args.gpu = gpu_backup
                self.args.reset = reset_backup

            if self.dist_env.is_distributed:
                torch.distributed.broadcast_object_list([self.arg_parser.to_dict()], src=0)
        else:
            a = [None]
            torch.distributed.broadcast_object_list(a, src=0)
            self.args = self.arg_parser.from_dict(a[0])

        self.use_tensorboard, self.use_wandb = get_plot_config(self.args)
        self.state = DotDict()
        self.state.iter = 0

        # Synchronize self.state before restoring
        state_saver = PyObjectSaver(self.state)
        if self.dist_env.is_master():
            if self.restore_pending:
                state_saver.load(self.restore_pending["state"])

            if self.dist_env.is_distributed:
                obj = state_saver.save()
                torch.distributed.broadcast_object_list([obj], src=0)
        elif self.dist_env.is_distributed:
            objl = [None]
            torch.distributed.broadcast_object_list(objl, src=0)
            state_saver.load(objl[0])

        self.run_invariants = {
            "args": self.arg_parser.to_dict()
        }

        if self.dist_env.is_master():
            constructor = plot.AsyncLogger if self.log_async else plot.Logger

            assert (not self.use_wandb) or (self.wandb_project_name is not None), \
                'Must specify wandb project name if logging to wandb.'

            assert self.args.name is not None or self.use_wandb, "Either name must be specified or W&B should be used"

            if self.restore_pending and self.restore_pending["run_invariants"]["wandb_id"]:
                wandb_args = {
                    "project": self.restore_pending["run_invariants"]["wandb_id"]["project"],
                    "id": self.restore_pending["run_invariants"]["wandb_id"]["run_id"],
                    "resume": "must"
                }
                if "entity" in self.restore_pending["run_invariants"]["wandb_id"]:
                    # Old checkpoints don't have entity, load only if available
                    wandb_args["entity"] = self.restore_pending["run_invariants"]["wandb_id"]["entity"]
            else:
                wandb_args = {
                    "project": self.wandb_project_name,
                    "config": self.arg_parser.to_dict()
                }

            self.summary = constructor(save_dir=os.path.join("save", self.args.name) if self.args.name is not None else None,
                                            use_tb=self.use_tensorboard,
                                            use_wandb=self.use_wandb,
                                            wandb_init_args=wandb_args,
                                            wandb_extra_config={
                                                "experiment_name": self.args.name,
                                                "n_nodes": self.dist_env.world_size or 1,
                                            },
                                            get_global_step = lambda: self.state.iter)

            self.run_invariants["wandb_id"] = self.summary.wandb_id
            if self.summary.wandb_id:
                self.wandb_project_name = self.summary.wandb_id["project"]

            if self.use_wandb:
                self.print_env_info()

            print(self.dist_env)

            self.create_dirs()
            self.save_startup_log()
            self.start_tensorboard()

        self.saver = Saver(self.dirs.checkpoint if self.dist_env.is_master() else None, self.args.save_interval,
                           keep_every_n_hours=None if self.use_wandb else 4)
        self.saver["state"] = self.state
        self.saver["run_invariants"] = deepcopy(self.run_invariants)

        self.setup_environment()

    @master
    def wait_for_termination(self):
        if self.args.keep_alive and self.use_tensorboard and not self.use_wandb:
            print("Done. Waiting for termination")
            while True:
                time.sleep(100)

    @master
    def save(self):
        if self.saver.last_saved_iter == self.state.iter:
            return

        res = self.saver.save(iter=self.state.iter)
        self.saver.cleanup()
        if res is not None:
            self.record_job_info()

    @master
    def tick(self):
        if self.saver.tick(iter=self.state.iter):
            self.record_job_info()

    @master
    def finish(self):
        self.summary.finish()
        if self.is_sweep or self.saver.last_saved_iter != self.state.iter:
            self.save()

        self.wait_for_termination()

    def to_device(self, data: Any) -> Any:
        return U.apply_to_tensors(data, lambda d: d.to(self.device))

    def distibute_model_weights(self):
        if self.dist_env.is_master():
            # if ditributed, send the full state to all workers
            if self.dist_env.is_distributed:
                # PyTorch bug: there is an int32 conversion in the distributed code that overflows if the data is
                # > 2G. So send it in pieces.
                for k, v in self.saver.get_data().items():
                    torch.distributed.broadcast_object_list([k, v], src=0)
        else:
            # if ditributed and worker, restore state form master
            ckpt = {}
            # Stich the pieces together
            for _ in range(len(self.saver.get_data())):
                a = [None, None]
                torch.distributed.broadcast_object_list(a, src=0)
                ckpt[a[0]] = a[1]

            ckpt = self.to_device(ckpt)
            self.saver.load_data(ckpt)

    def restore(self):
        if self.dist_env.is_master():
            if self.restore_pending is not None:
                assert self.saver.load_data(self.restore_pending), "Restoring failed."
                self.restore_pending = None
                restored = True
            else:
                restored = self.saver.load()

            if restored:
                # Do not restore these things
                self.saver.register("run_invariants", deepcopy(self.run_invariants), replace=True)
            else:
                if self.dist_env.is_preemtible() and not self.dist_env.is_restart():
                    # Save initial state if preemtible, such that the args can be loaded later.
                    self.save()

        self.distibute_model_weights()

    def get_storage_path(self, path: str) -> str:
        assert self.dist_env.is_master()
        path = os.path.join(self.dirs.export, path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    @master
    def export_tensor(self, rel_path: str, data: Union[torch.Tensor, np.ndarray]):
        data = U.apply_to_tensors(data, lambda x: x.detach().cpu().numpy())
        torch.save(data, self.get_storage_path(rel_path + ".pth"))

    def fix_names(self, plotlist: Dict[str, Any]) -> Dict[str, Any]:
        def fix_name(s: str) -> str:
            s = self.find_slash.sub('/', s)
            s = self.remove_firstlast_slash.sub('', s)
            return s

        return {fix_name(k): v for k, v in plotlist.items()}

    @master
    def log(self, plotlist, step=None):
        if self.args.wandb_bug_workaround and self.use_wandb:
            filtered = {k: v for k, v in plotlist.items() if not isinstance(v, visualize.plot.TextTable)}
            if len(filtered) != len(plotlist) and not self.wandb_bug_found:
                print("WARNING: wandb_bug_workaround enabled. Refusing to log tables")
                self.wandb_bug_found = True
            plotlist = filtered

        plotlist = self.fix_names(plotlist)
        if plotlist:
            self.summary.log(plotlist, step)
