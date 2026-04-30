from ..utils.distributed_ops import reduce_any
from ..interfaces import Result
import torch
import torch.utils.data
from tqdm import tqdm
from typing import Dict, Any, Iterable, Tuple, Optional
from dataclasses import dataclass
import torch.distributed
import time
from ..layers import logging_layer
from .task_db import args
from ..import helpers, data_structures, utils, loader, optimizer

@dataclass
class LastBestMarker:
    iter: int
    loss: float
    accuracy: float

@args
def a(parser: helpers.ArgumentParser):
    parser.add_argument("-batch_size", default=128)
    parser.add_argument("-lr", default=1e-3)
    parser.add_argument("-min_lr_multiplier", default=0.0)
    parser.add_argument("-wd", default=0.0)
    parser.add_argument("-lr_warmup", default=0)
    parser.add_argument("-test_interval", default=1000)
    parser.add_argument("-n_microbatch", default="none", parser=parser.int_or_none_parser)
    parser.add_argument("-per_device_batch_size", default="none", parser=parser.int_or_none_parser)
    parser.add_argument("-lr_sched.steps", default="", parser=parser.int_list_parser)
    parser.add_argument("-lr_sched.gamma", default=0.1)
    parser.add_argument("-lr_sched.type", default="step", choice=["step", "noam", "cos"])
    parser.add_argument("-length_bucketed_sampling", default=False)
    parser.add_argument("-grad_clip", default="1.0", parser=parser.float_or_none_parser)
    parser.add_argument("-test_batch_size", default="None", parser=parser.int_or_none_parser)
    parser.add_argument("-val_log_details", default=False)


class Task:
    valid_loaders: data_structures.DotDict
    batch_dim: int
    TRAIN_NUM_WORKERS = 1
    VALID_NUM_WORKERS = 1
    IGNORE_INDEX = 0

    def __init__(self, helper: helpers.TrainingHelper):
        self.helper = helper
        self.training = True
        self.helper.state.best_losses = {}
        self.helper.state.best_accuracies = {}
        self.valid_sets = data_structures.DotDict()
        self.loss_average = utils.Average()
        self.forward_time_meter = utils.ElapsedTimeMeter()
        self.load_time_meter = utils.ElapsedTimeMeter()
        self.plot_time_meter = utils.ElapsedTimeMeter()
        self.total_n_token_in_period = 0
        self.last_token_measure_time = time.time()
        self.models = {}

    def add_model(self, name: str, module: torch.nn.Module):
        self.models[name] = module
        setattr(self, name, module)

    def create_lr_scheduler(self):
        if self.helper.args.lr_sched.type == "step":
            self.lr_scheduler = optimizer.StepLrSched(self.helper.args.lr, self.helper.args.lr_sched.steps,
                                                      self.helper.args.lr_sched.gamma)

        elif self.helper.args.lr_sched.type == "noam":
            self.lr_scheduler = optimizer.NoamLRSched(self.helper.args.lr, self.helper.args.state_size,
                                                      self.helper.args.lr_warmup)
        elif self.helper.args.lr_sched.type == "cos":
            if self.helper.args.stop_after is None:
                raise ValueError("Cosine annealing requires stop_after to be set")
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, self.helper.args.stop_after,
                eta_min=self.helper.args.min_lr_multiplier*self.helper.args.lr)
        else:
            assert False

    def use_length_bucketed(self, vset: torch.utils.data.Dataset) -> bool:
        return "in_len" in vset[0]

    def create_valid_loader(self, vset: torch.utils.data.Dataset) -> torch.utils.data.DataLoader:
        # Do bucketed testing even when the bucketed training is not enabled
        if self.use_length_bucketed(vset):
            batch_size = self.test_batch_size
            batch_sampler = loader.sampler.BucketedSampler(vset, batch_size, infinite=False, long_first=True,
                                                                     random_order=False,
                                                                     world_size=self.helper.dist_env.world_size,
                                                                     rank=self.helper.dist_env.rank,
                                                                     seed=0)
            batch_size = 1
        else:
            batch_size = self.helper.get_batch_size(self.test_batch_size)
            if self.helper.dist_env.is_distributed:
                vset = loader.DatasetSplitter(vset, self.helper.dist_env.world_size, self.helper.dist_env.rank)
            batch_sampler = None

        return torch.utils.data.DataLoader(vset, batch_size=batch_size, batch_sampler=batch_sampler,
                                   collate_fn=loader.collate.VarLengthCollate(batch_dim=self.batch_dim),
                                   num_workers=self.VALID_NUM_WORKERS, persistent_workers=self.VALID_NUM_WORKERS > 0)


    def create_loaders(self):
        self.train_loader = self.create_train_loader(self.train_set, mask = False)
        self.valid_loaders = data_structures.DotDict()
        self.valid_loaders.update({k: self.create_valid_loader(v) for k, v in self.valid_sets.items()})

    def replace_valid_set(self, name: str, vset: torch.utils.data.Dataset):
        self.valid_sets[name] = vset
        self.valid_loaders[name] = self.create_valid_loader(vset)

    def create_train_loader_bs(self, ldr: torch.utils.data.Dataset, batch_size: int, seed: Optional[int] = None) \
                            -> torch.utils.data.DataLoader:

        if self.helper.args.length_bucketed_sampling and self.use_length_bucketed(ldr):
            batch_sampler = loader.sampler.BucketedSampler(ldr, batch_size, infinite=True, drop_last=True,
                                                                     random_order=True,
                                                                     world_size=self.helper.dist_env.world_size,
                                                                     rank=self.helper.dist_env.rank,
                                                                     seed=0)
            sampler = None
            batch_size = 1
        else:
            batch_size = self.helper.get_batch_size(batch_size)
            if self.helper.dist_env.is_distributed:
                ldr = loader.DatasetSplitter(ldr, self.helper.dist_env.world_size,
                                                          self.helper.dist_env.rank)

            batch_sampler = None
            sampler = loader.sampler.InfiniteSampler(ldr, seed = seed)

        return torch.utils.data.DataLoader(ldr, batch_size=batch_size,
                                           sampler=sampler, batch_sampler=batch_sampler,
                                           collate_fn=loader.collate.VarLengthCollate(
                                               batch_dim=self.batch_dim, ignore_symbol=self.IGNORE_INDEX),
                                           num_workers=self.TRAIN_NUM_WORKERS, pin_memory=True,
                                           persistent_workers=self.TRAIN_NUM_WORKERS > 0)

    def create_validate_on_train(self, set: torch.utils.data.Dataset):
        self.valid_sets.train = set

        if self.helper.dist_env.is_distributed:
            set = loader.DatasetSplitter(set, self.helper.dist_env.world_size,
                                                      self.helper.dist_env.rank)

        self.valid_loaders.train = torch.utils.data.DataLoader(set, batch_size=self.helper.get_batch_size(),
                                   collate_fn=loader.collate.VarLengthCollate(batch_dim=self.batch_dim,
                                        ignore_symbol=self.IGNORE_INDEX),
                                   sampler=loader.sampler.SubsetSampler(set, (len(self.valid_sets.iid)
                                        if "iid" in self.valid_sets else 1000) // self.helper.dist_env.world_size),
                                   num_workers=self.VALID_NUM_WORKERS, persistent_workers=self.VALID_NUM_WORKERS > 0)

    def clip_gradients(self):
        if self.helper.args.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.helper.args.grad_clip)

    def set_optimizer_lr(self, lr: float):
        utils.set_lr(self.optimizer, lr)

    def set_linear_warmup(self, curr_step: int, n_steps: int, final: float) -> float:
        if curr_step >= n_steps:
            lr = final
        else:
            lr = final / n_steps * (curr_step+1)

        self.set_optimizer_lr(lr)
        return lr

    def set_lr(self):
        has_builtin_warmup = self.helper.args.lr_sched.type in {"noam"}
        offset = 0 if has_builtin_warmup else self.helper.args.lr_warmup

        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.LRScheduler):
            self.lr_scheduler.step(max(0, self.helper.state.iter - offset))
        else:
            self.set_optimizer_lr(self.lr_scheduler.get(max(0, self.helper.state.iter - offset)))

        if self.helper.args.lr_warmup > self.helper.state.iter and not has_builtin_warmup:
            self.set_linear_warmup(self.helper.state.iter, self.helper.args.lr_warmup,
                                   utils.get_lr(self.optimizer))

        if self.helper.state.iter % 100 == 0:
            self.helper.log({"lr": utils.get_lr(self.optimizer)})

    def prepare_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self.helper.to_device(data)

    def get_batch_size(self, data: Dict[str, Any]) -> int:
        for v in data.values():
            if torch.is_tensor(v) and v.ndim > self.batch_dim:
                return v.shape[self.batch_dim]

        raise ValueError("Unable to automatically determine the local batch size.")

    def run_model(self, data: Dict[str, torch.Tensor], ubatch: int = 0) -> Tuple[Result, Dict[str, Any]]:
        return self.model_interface(data, self.helper.state.iter, ubatch)

    def run_model_validation(self, data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Any]:
        res, _ = self.run_model(data)
        return res.loss, self.model_interface.decode_outputs(res)

    def set_eval(self):
        self.training = False
        for m in self.models.values():
            m.eval()

    def set_train(self):
        self.training = True
        for m in self.models.values():
            m.train()

    def validate_on(self, set: torch.utils.data.Dataset, loader: torch.utils.data.DataLoader) -> Tuple[Any, float]:
        self.set_eval()

        bucketed = self.use_length_bucketed(set)
        with torch.no_grad():
            loss_sum = 0
            count = 0

            test = set.start_test()
            l = len(loader)
            lmax = l

            # If distributed testing is not supported by the test, copy everything to the master process
            dist_test = torch.distributed.is_initialized() and not (hasattr(test, "SUPPORTS_DISTRIBUTED") and test.SUPPORTS_DISTRIBUTED)

            if dist_test:
                lmax = torch.tensor(lmax, dtype=torch.int32).cuda()
                torch.distributed.all_reduce(lmax, torch.distributed.ReduceOp.MAX)
                lmax = lmax.item()

            for d in tqdm(loader):
                d = self.prepare_data(d)
                loss, output = self.run_model_validation(d)
                batch_size = self.get_batch_size(d)

                this_loss = loss.sum().item() * batch_size

                if dist_test:
                    # If the dataset doesn't support distributed testing, we need to gather the results
                    alist = [None] * self.helper.dist_env.world_size
                    torch.distributed.all_gather_object(alist, (output, d, this_loss, batch_size))

                    for output, d, this_loss, bs in alist:
                        if output is None:
                            continue
                        loss_sum += this_loss
                        count += bs
                        test.step(output, d)
                else:
                    loss_sum += this_loss
                    count += batch_size
                    test.step(output, d)

            if dist_test:
                for _ in range(lmax - l):
                    # if the work is not even for all workers, send dummy messages around to not get blocked
                    alist = [None] * self.helper.dist_env.world_size
                    torch.distributed.all_gather_object(alist, (None, None, None, None))

        print(f"Validation done on worker {self.helper.dist_env.rank}.")
        self.set_train()

        return test, reduce_any(loss_sum) / reduce_any(count)

    def validate_on_name(self, name: str) -> Tuple[Any, float]:
        print(f"Starting validation on {name}...")
        return self.validate_on(self.valid_sets[name], self.valid_loaders[name])

    def fix_loaded_best_losses(self):
        # Loading destroys the class.
        to_fix = [self.helper.state.best_losses, self.helper.state.best_accuracies]
        for f in to_fix:
            for k, v in f.items():
                if isinstance(v, dict):
                    new_v = LastBestMarker(0, 0, 0)
                    new_v.__dict__.update(v)
                    f[k] = new_v

    def update_best_accuracies(self, name: str, accuracy: float, loss: float):
        self.fix_loaded_best_losses()

        if name not in self.helper.state.best_losses or loss < self.helper.state.best_losses[name].loss:
                self.helper.state.best_losses[name] = LastBestMarker(self.helper.state.iter, loss, accuracy)

        if name not in self.helper.state.best_accuracies or accuracy > \
                self.helper.state.best_accuracies[name].accuracy:
            self.helper.state.best_accuracies[name] = LastBestMarker(self.helper.state.iter, loss, accuracy)

        return {
            f"{name}/time_since_best_loss": self.helper.state.iter - self.helper.state.best_losses[name].iter,
            f"{name}/time_since_best_accuracy": self.helper.state.iter - self.helper.state.best_accuracies[name].iter
        }

    def validate_on_names(self, name_it: Iterable[str]) -> Dict[str, Any]:
        charts = {}
        sum_accuracy = 0
        sum_all_losses = 0

        logging_layer.get_logs(self.model)

        for name in name_it:
            test, loss = self.validate_on_name(name)

            if self.helper.args.dump_logs:
                logging_layer.dump_logs(self.model, self.helper.get_storage_path("log_dumps") + f"/{self.helper.state.iter}/valid/{name}/")
            logs = logging_layer.get_logs(self.model)

            print(f"Validation accuracy on {name}: {test.accuracy}")
            charts[f"{name}/loss"] = loss
            sum_all_losses += loss
            charts.update({f"{name}/{k}": v for k, v in test.plot().items()})
            if self.helper.args.val_log_details:
                charts.update({f"{name}/{k}": v for k, v in logs.items()})
            sum_accuracy += test.accuracy

            charts.update(self.update_best_accuracies(name, test.accuracy, loss))

        charts["mean_accuracy"] = sum_accuracy / max(len(self.valid_sets), 1)
        charts["mean_loss"] = sum_all_losses / max(len(self.valid_sets), 1)
        return charts

    def validate(self) -> Dict[str, Any]:
        return self.validate_on_names(self.valid_sets.keys())

    def plot(self, res: Result) -> Dict[str, Any]:
        plots = {}

        self.loss_average.add(res.loss)

        if self.helper.state.iter % 200 == 0:
            plots.update(res.plot())

        if self.helper.state.iter % 20 == 0:
            if self.total_n_token_in_period:
                now = time.time()
                plots["timing/total_ms_per_token"] = (now - self.last_token_measure_time)*1000/ \
                                                                     (20*self.total_n_token_in_period)
                plots["timing/ms_per_token"] = self.forward_time_meter.get(False)*1000/ \
                                                                     (20*self.total_n_token_in_period)
                self.total_n_token_in_period = 0
                self.last_token_measure_time = now

            plots["train/loss"] = self.loss_average.get()
            plots["timing/ms_per_iter"] = self.forward_time_meter.get(True)*1000/20
            plots["timing/ms_per_load"] = self.load_time_meter.get(True)*1000/20
            plots["timing/ms_per_plot"] = self.plot_time_meter.get(True)*1000/20

        if (self.helper.state.iter % self.helper.args.test_interval == 0) or (self.helper.state.iter == self.helper.args.stop_after):
            plots.update({f"validation/{k}": v for k, v in self.validate().items()})

        return plots

    def create_model_interface(self):
        raise NotImplementedError()

    def create_datasets(self):
        raise NotImplementedError()

    def create_model(self) -> torch.nn.Module:
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()

    def save_weights(self):
        self.helper.save()

    def load_weights(self, file_path: str):
        old_model = torch.load(file_path)
        self.model.load_state_dict(old_model["model"])

    def finish(self):
        self.helper.save()

    @property
    def n_microbatch(self) -> Optional[int]:
        if self.helper.args.n_microbatch is not None:
            if self.helper.args.per_device_batch_size is not None:
                raise ValueError("Both n_microbatch and per_device_batch_size are set.")
            return self.helper.args.n_microbatch

        if not self.helper.args.per_device_batch_size:
            return None

        per_dev_bs = (self.helper.args.batch_size + self.helper.dist_env.world_size - 1) // self.helper.dist_env.world_size
        return (per_dev_bs + self.helper.args.per_device_batch_size - 1) // self.helper.args.per_device_batch_size

    @property
    def test_batch_size(self) -> int:
        if self.helper.args.test_batch_size is not None:
            return self.helper.args.test_batch_size

        if self.n_microbatch is not None:
            return self.helper.args.batch_size // self.n_microbatch

        return self.helper.args.batch_size