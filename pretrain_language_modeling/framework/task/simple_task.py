import torch
import torch.nn
import torch.optim
import torch.utils.data
import torch.cuda.amp
from tqdm import tqdm
import time
from typing import Optional, Dict, Any, Tuple, List
from ..interfaces.result import LossOnlyResult, Result
from .task import Task
from .task_db import args
from ..layers.layer_with_stats import LayerStatProcessor
from ..layers.regularized_layer import LayerRegularizer
from ..layers.layer_with_visualization import LayerVisualizer
import torch.distributed
from ..layers.logging_layer import get_logs, dump_logs
from ..layers.once_per_iter_layer import call_post_iter, call_pre_iter, call_before_loss
from ..utils import U
from .. import helpers, utils, data_structures


@args
def a(parser: helpers.ArgumentParser):
    parser.add_argument("-reg_scales", default="", parser=parser.float_params_parser)
    parser.add_argument("-reg_lin_decay", default="", parser=parser.str_list_parser)
    parser.add_argument("-reg", default=1.0)
    parser.add_argument("-optimizer", default="adamw", choice=["adam", "adamw", "sgd", "adagrad"])
    parser.add_argument("-adam.betas", default="0.9,0.999", parser=parser.float_list_parser)
    parser.add_argument("-adam.eps", default=1e-8)
    parser.add_argument("-stop_after", default="None", parser=parser.int_or_none_parser)
    parser.add_argument("-amp", default=False)
    parser.add_argument("-bfloat16", default=True)
    parser.add_argument("-nan_detect", default=False)
    parser.add_argument("-max_length_per_batch", default="none", parser=parser.int_or_none_parser)
    parser.add_argument("-log_grad_norms", default=False)
    parser.add_argument("-nan_detect", default=False)
    parser.add_argument("-speedtest", default="none", choice=["none", "iter"])
    parser.add_argument("-dump_logs", default=False)
    parser.add_argument("-debug_plot_interval", default="none", parser=parser.int_or_none_parser)


class SimpleTask(Task):
    MAX_LENGHT_PER_BATCH = None
    NO_OUTPUT_TRACKING = False
    train_set: torch.utils.data.Dataset
    train_loader: torch.utils.data.DataLoader
    model: torch.nn.Module

    def create_datasets(self):
        raise NotImplementedError()

    def create_model_interface(self):
        pass

    def create_model(self) -> torch.nn.Module:
        raise NotImplementedError()

    def create_state(self):
        pass

    @property
    def amp_enabled(self):
        return torch.cuda.is_available() and self.helper.args.amp

    @property
    def time_dim(self) -> int:
        return 1 - self.batch_dim

    def __init__(self, helper: helpers.TrainingHelper):
        super().__init__(helper)

        self.fetcher = None

        self.bf16_enabled = self.helper.args.bfloat16 and torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8 and torch.cuda.is_bf16_supported()

        self.avg_num_chunks = utils.Average()
        self.reg_loss_average = utils.DictAverage()
        self.max_grad = 0
        self.time_sum = 0

        self.create_datasets()
        self.create_loaders()
        model = self.create_model()
        self.add_model("model", model)
        for n, m in {n: m.to(self.helper.device) for n, m in self.models.items()}.items():
            self.add_model(n, m)
        self.set_train()

        # self.compiled = False

        # self.model = torch.compile(self.model)

        # if self.helper.dist_env.is_distributed:
        #     self.grad_syncer = GradSyncer(self.model)

        self.stat_processor = LayerStatProcessor(self.model)

        self.create_model_interface()
        self.create_optimizer()
        self.create_lr_scheduler()

        self.regularizer = LayerRegularizer(
            list(self.models.values()), self.helper.args.stop_after, self.helper.args.reg_scales,
            self.helper.args.reg_lin_decay)

        if self.amp_enabled and self.bf16_enabled:
            print("Training in bfloat16...")

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled and not self.bf16_enabled)
        self.helper.saver["scaler"] = self.scaler

        n_params = sum(sum(p.numel() for p in model.parameters()) for model in self.models.values())
        print(f"Total number of model parameters: {n_params}")

        for n, mod in self.models.items():
            self.helper.saver[n] = mod

        self.visualizer = LayerVisualizer(list(self.models.values()))

        self.create_state()
        self.helper.restore()

        self.helper.log({"n_params": n_params})

        if self.helper.args.nan_detect:
            torch.autograd.set_detect_anomaly(True)

            # based on https://discuss.pytorch.org/t/finding-source-of-nan-in-forward-pass/51153/3
            def nan_hook(self, inp, output):
                if not isinstance(output, tuple):
                    outputs = [output]
                else:
                    outputs = output

                for i, out in enumerate(outputs):
                    def detect(out):
                        nan_mask = ~torch.isfinite(out)
                        if nan_mask.any():
                            print("In", self.__class__.__name__)
                            raise RuntimeError(f"Found non-finite in output {i} at indices: ", nan_mask.nonzero(), "where:", out[nan_mask.nonzero()[:, 0].unique(sorted=True)])

                    U.apply_recursive(out, detect, torch.is_tensor)

            for model in self.models.values():
                for submodule in model.modules():
                    submodule.register_forward_hook(nan_hook)

    def fetch_thread(self):
        data = self.prepare_data(self.get_train_batch())
        n_chunks = self.get_n_chunks(data)
        d_chunks = self.chunk_batch_dim(data, n_chunks)

        return data, d_chunks

    def create_train_loader(self, loader: torch.utils.data.Dataset, seed: Optional[int] = None,
                            batch_size: Optional[int] = None) -> torch.utils.data.DataLoader:

        return super().create_train_loader_bs(loader, batch_size or self.helper.args.batch_size, seed)

    def create_data_fetcher(self):
        if self.fetcher is not None:
            self.fetcher.finish()

        self.data_iter = iter(self.train_loader)
        self.fetcher = helpers.StoppingParallelProducer(self.fetch_thread)

    def set_train_set(self, ds: torch.utils.data.Dataset, seed: Optional[int] = None):
        self.train_set = ds

        fetcher_exists = self.fetcher is not None
        if fetcher_exists:
            # Ensure we won't start fetching from the new train set
            self.fetcher.finish()
            self.fetcher = None

        self.train_loader = self.create_train_loader(self.train_set, seed)

        if fetcher_exists:
            self.create_data_fetcher()

    def create_loaders(self):
        self.train_loader = self.create_train_loader(self.train_set)
        self.valid_loaders = data_structures.DotDict()
        self.valid_loaders.update({k: self.create_valid_loader(v) for k, v in self.valid_sets.items()})

    def get_optimizer_param_list(self):
        return self.model.parameters()

    def create_optimizer(self):
        if self.helper.args.optimizer in ["adam", "adamw"]:
            opt = torch.optim.Adam if self.helper.args.optimizer == "adam" else torch.optim.AdamW
            self.set_optimizer(opt(self.get_optimizer_param_list(), self.helper.args.lr,
                                                weight_decay=self.helper.args.wd, betas=self.helper.args.adam.betas,
                                                eps=self.helper.args.adam.eps))
        elif self.helper.args.optimizer == "adagrad":
            self.set_optimizer(torch.optim.Adagrad(self.get_optimizer_param_list(), self.helper.args.lr,
                                                    weight_decay=self.helper.args.wd))
        elif self.helper.args.optimizer == "sgd":
            self.set_optimizer(torch.optim.SGD(self.get_optimizer_param_list(), self.helper.args.lr,
                                               weight_decay=self.helper.args.wd, momentum=0.9))
        else:
            assert False, f"Unsupported optimizer: {self.helper.args.optimizer}"

    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        self.helper.saver.register("optimizer", self.optimizer, replace=True)

    def get_train_batch(self) -> Dict[str, Any]:
        return next(self.data_iter)

    def chunk_batch_dim(self, data: Dict[str, Any], n: int) -> List[Dict[str, Any]]:
        if n == 1:
            return [data]

        res = [{} for _ in range(n)]
        for k, v in data.items():
            assert torch.is_tensor(v), "Only tensors are supported by autosplitting"

            bd = self.batch_dim if self.batch_dim < v.ndimension() else 0
            # assert v.shape[bd] % n == 0, f"Batch (dim {bd} of input {k} of shape {v.shape} is not divisible by {n})"

            for i, c in enumerate(v.chunk(n, dim=bd)):
                res[i][k] = c

        # Avoid unnecessary computation.
        if "in" in data and "in_len" in data:
            for r in res:
                r["in"] = r["in"].narrow(1 - self.batch_dim, 0, int(r["in_len"].max().item()))

        if "out" in data and "out_len" in data and data["out"].ndim > 1:
            for r in res:
                r["out"] = r["out"].narrow(1 - self.batch_dim, 0, int(r["out_len"].max().item()))

        return res

    def is_seq2seq_task(self, data: Dict[str, Any]) -> bool:
        return "in_len" in data and "out_len" in data

    def get_seq_length(self, data: Dict[str, Any]) -> int:
        # This assumes separate encoder and decoder
        return max(data["in"].shape[self.time_dim], data["out"].shape[self.time_dim] if data["out"].ndim > 1 else 0)

    def get_n_chunks(self, data: Dict[str, Any]) -> int:
        if self.n_microbatch:
            return self.n_microbatch

        max_length_per_batch = self.helper.args.max_length_per_batch or self.MAX_LENGHT_PER_BATCH
        if self.is_seq2seq_task(data) and max_length_per_batch:
            # The formula below assumes quadratic memory consumption
            return int(2**int(self.get_seq_length(data) / max_length_per_batch))
        return 1

    def post_backward(self) -> Dict[str, Any]:
        return {}

    def get_regularizers(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.regularizer.get(self.helper.state.iter)

    def run_ubatch(self, data: Dict[str, Any], ubatch: int, total_batch_size: int) -> Tuple[Result, float, Dict[str, Any]]:
        plots = {}

        plot_now = self.training and ubatch==0 and self.helper.args.debug_plot_interval and \
                   self.helper.state.iter % self.helper.args.debug_plot_interval == 0

        if plot_now:
            self.visualizer.prepare()

        with torch.cuda.amp.autocast(enabled=self.amp_enabled, dtype=torch.bfloat16 if self.bf16_enabled else None):
            res, custom_plots = self.run_model(data, ubatch)
            call_before_loss(self.model)
            if ubatch == 0:
                plots.update(custom_plots)

        if plot_now:
            plots.update(self.visualizer.plot())

        # weights for microbatch accumulation
        weight = self.get_batch_size(data) / total_batch_size
        reg_loss, reg_log = self.get_regularizers()
        self.reg_loss_average.add(reg_log)
        total_loss = (res.loss + reg_loss * self.helper.args.reg) * self.helper.get_loss_scaling()

        if self.NO_OUTPUT_TRACKING:
            res = LossOnlyResult(res.loss.detach())
        else:
            res = res.detach()

        if not torch.isfinite(total_loss):
            for model in self.models.values():
                for n, p in model.named_parameters():
                    if not torch.isfinite(p).all():
                        print(f"Found non-finite weight {n}")

                for n, p in model.named_buffers():
                    if not torch.isfinite(p).all():
                        print(f"Found non-finite buffer {n}")

            assert False, f"Loss not finite ({total_loss})"

        self.scaler.scale(total_loss * weight).backward()

        pbwout = self.post_backward()
        if ubatch == 0:
            plots.update(pbwout)

        return res, weight, plots

    def prepare_visualizer(self, data: Dict[str, Any]):
        self.visualizer.prepare()

    def train_step(self) -> Tuple[Result, Dict[str, Any]]:
        plots = {}

        if self.helper.args.speedtest=="iter":
            torch.cuda.synchronize()

        with self.forward_time_meter:
            self.set_lr()
            self.optimizer.zero_grad(set_to_none=True)

            data, d_chunks = self.fetcher.get()

            res_list = []
            weights = []

            self.avg_num_chunks.add(len(d_chunks))

            total_batch_size = self.get_batch_size(data)

            profiler = None
            # if self.helper.state.iter == 3:
            #     profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True, with_stack=True, profile_memory=True)
            #     profiler.__enter__()


            call_pre_iter(self.model)
            for ubatch, d in enumerate(d_chunks):
                res, weight, p = self.run_ubatch(d, ubatch, total_batch_size)
                res_list.append(res)
                weights.append(weight)
                plots.update(p)

            if self.helper.dist_env.is_distributed:
                aops = []
                for model in self.models.values():
                    for p in model.parameters():
                        if p.grad is None:
                            continue
                        p.grad = p.grad.contiguous()
                        aops.append(torch.distributed.all_reduce(p.grad, async_op=True))

                    for a in aops:
                        a.wait()

            call_post_iter(self.model)

            self.scaler.unscale_(self.optimizer)

            if self.helper.args.grad_clip:
                gn = torch.nn.utils.clip_grad_norm_(
                    [p for model in self.models.values() for p in model.parameters() if p.grad is not None],
                    self.helper.args.grad_clip)
                self.max_grad = max(self.max_grad, gn)

            if self.helper.args.log_grad_norms:
                for mn, mod in self.models.items():
                    for n, p in mod.named_parameters():
                        if p.grad is not None:
                            plots[f"{mn}/grad_norms/{n}"] = p.grad.detach().norm().item()


            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.helper.state.iter += 1

            res = res_list[0].__class__.merge(res_list, weights) if len(res_list) > 1 else res_list[0]

            if self.helper.args.speedtest in {"iter"}:
                torch.cuda.synchronize()

            if profiler is not None:
                profiler.__exit__(None, None, None)
                profiler.export_chrome_trace("trace_all.json")
                assert False


            # if self.helper.state.iter % 20 == 0:

        if "in_len" in data:
            n_total_tokens = (data["in_len"] + data["out_len"]).sum()
            if self.helper.dist_env.is_distributed:
                torch.distributed.all_reduce(n_total_tokens)

            self.total_n_token_in_period += n_total_tokens

        return res, plots

    def plot(self, res: Result) -> Dict[str, Any]:
        res = super().plot(res)

        if self.helper.args.dump_logs and self.helper.dist_env.is_master():
            dump_logs(self.model, self.helper.get_storage_path("log_dumps") + f"/{self.helper.state.iter}")

        if self.helper.state.iter % 20 == 1:
            if len(self.models) > 1:
                for mn, m in self.models.items():
                    res.update({f"{mn}/{k}": v for k, v in get_logs(m).items()})
            else:
                res.update(get_logs(self.model))

            res["average_num_chunks"] = self.avg_num_chunks.get()
            for k, v in self.reg_loss_average.get().items():
                res[f"train/reg_loss/{k}"] = v

            if self.helper.args.grad_clip:
                res["max_grad"] = self.max_grad
                self.max_grad = 0

            res.update({f"layer_stats/{k}": v for k, v in self.stat_processor.get().items()})

        return res

    def train(self):
        self.loss_average.reset()

        self.data_iter = iter(self.train_loader)
        self.fetcher = helpers.StoppingParallelProducer(self.fetch_thread)

        # try:
        #     while (self.helper.args.stop_after or 10e10) > self.helper.state.iter:
        #         self.load_time_meter.stop()

        #         res, plots = self.train_step()
        #         plots.update(self.plot(res))

        #         with self.plot_time_meter:
        #             self.helper.log(plots)

        #         self.load_time_meter.start()

        #         self.helper.tick()
        # except self.fetcher.Stopped:
        #     pass

        max_iters = self.helper.args.stop_after or int(10e10)
        # breakpoint()

        with tqdm(total=max_iters, desc="Training Progress", unit="step") as pbar:
            while max_iters > self.helper.state.iter:
                self.load_time_meter.stop()

                res, plots = self.train_step()

                plots.update(self.plot(res))
                with self.plot_time_meter:
                    self.helper.log(plots, step=self.helper.state.iter)
                self.load_time_meter.start()

                self.helper.tick()
                pbar.update(1)
        self.helper.saver.end_time = time.time()