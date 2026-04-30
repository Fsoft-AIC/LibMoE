import framework
import torch
import torch.nn
import torch.utils.data
from framework import dataset
from typing import Tuple, Any, Dict, List, Optional
from interfaces import LanguageModelInterface
from framework.task import task, args, SimpleTask
import random
from .transformer_lm_mixin import TransformerLMMixin


@args
def a(parser: framework.helpers.ArgumentParser):
    parser.add_argument("-lm.state_drop_probability", default=0.0)
    parser.add_argument("-lm.unroll", default=100)
    parser.add_argument("-lm.unroll_eval", default="none", parser=parser.int_or_none_parser)
    parser.add_argument("-lm.example_context", default=100)
    parser.add_argument("-lm.example_window", default=40)


@task()
class LMBase(TransformerLMMixin, SimpleTask):
    VALID_NUM_WORKERS = 1
    TRAIN_NUM_WORKERS = 2

    def __init__(self, helper: framework.helpers.TrainingHelper):
        super().__init__(helper)

        self.rnd_valid = {k: self.pick_random_sentences(v, 3, self.helper.args.lm.example_context,
                             self.helper.args.lm.example_window) for k, v in self.valid_sets.items() if hasattr(v, "linear_len")}
        self.rnd_train = self.pick_random_sentences(self.train_set, 3, self.helper.args.lm.example_context,
                                                    self.helper.args.lm.example_window) if hasattr(self.train_set, "linear_len") else None

    def create_state(self):
        self.helper.state.epoch = 0

    def create_model_interface(self):
        self.model_interface = LanguageModelInterface(
            self.model, drop_state_prob=self.helper.args.lm.state_drop_probability, dist_env=self.helper.dist_env,
            n_ubatches=self.n_microbatch)
        self.helper.saver["interface"] = self.model_interface

    def validate_on(self, set: torch.utils.data.Dataset, loader: torch.utils.data.DataLoader) -> Tuple[Any, float]:
        state = self.model_interface.state
        self.model_interface.reset_state()
        res = super().validate_on(set, loader)
        self.model_interface.state = state
        return res

    def log_epoch(self):
        self.helper.log({"epoch": self.helper.state.epoch})

    def start_next_epoch(self):
        self.model_interface.reset_state()
        self.helper.state.epoch += 1
        self.log_epoch()

    def get_train_batch(self) -> Dict[str, Any]:
        try:
            return next(self.data_iter)
        except StopIteration:
            self.start_next_epoch()
            self.data_iter = iter(self.train_loader)
            return next(self.data_iter)

    def create_sampler(self, loader: torch.utils.data.Dataset, batch_size: int, allow_uneven: bool = False) -> \
                       framework.loader.sampler.MultibatchSequentialSampler:

        return framework.loader.sampler.MultibatchSequentialSampler(loader, batch_size,
                            world_size=self.helper.dist_env.world_size, rank=self.helper.dist_env.rank,
                            allow_uneven=allow_uneven)

    def unroll_eval(self) -> int:
        return self.helper.args.lm.unroll_eval or self.helper.args.lm.unroll

    def create_valid_loader(self, vset: torch.utils.data.Dataset,
                            batch_size: Optional[int] = None) -> torch.utils.data.DataLoader:

        if batch_size is None:
            batch_size = self.test_batch_size

        return torch.utils.data.DataLoader(vset,
                                   batch_sampler=self.create_sampler(vset, batch_size, allow_uneven=False),
                                   collate_fn=framework.loader.collate.VarLengthCollate(
                                        batch_dim=self.batch_dim),
                                   num_workers=self.VALID_NUM_WORKERS)

    def create_train_loader(self, loader: torch.utils.data.Dataset) -> torch.utils.data.DataLoader:
        sampler = self.create_sampler(loader, self.helper.args.batch_size)
        self.helper.saver.register("sampler", sampler, replace=True)

        return torch.utils.data.DataLoader(loader, batch_sampler=sampler, num_workers=self.TRAIN_NUM_WORKERS,
                                           pin_memory=True, collate_fn=framework.loader.collate.VarLengthCollate(
                                           batch_dim=self.batch_dim))

    def create_datasets(self):
        self.batch_dim = 1
        self.train_set = dataset.Enwik8("train", self.helper.args.lm.unroll)
        self.valid_sets.val = dataset.Enwik8("valid", self.unroll_eval())
        self.valid_sets.test = dataset.Enwik8("test", self.unroll_eval())

    def train(self):
        self.log_epoch()
        super().train()

    def pick_random_sentences(self, set, n: int, content_length: int, window: int) -> Tuple[torch.Tensor, List[str]]:
        res = []

        window_offset = max(0, window - content_length)
        for i in range(n):
            pos = random.randrange(window, set.linear_len() - max(content_length, window))
            res.append(set.get_linear(pos + window_offset, max(content_length, window) + window))

        data = torch.tensor(res, dtype=torch.long, device=self.helper.device).T
        input = data[abs(window_offset) : -window]

        to_display = data[abs(window_offset) + content_length - window:]
        to_display_str = []
        for b in range(to_display.shape[1]):
            to_display_str.append(set.vocabulary.to_string(to_display[:, b].cpu().numpy().tolist()))

        return input, to_display_str

    def mark_token(self, text: str, pos: int) -> str:
        # Pos is unaccurate for byte dataset because of the bytearray->string conversion
        return text

    def generate_example(self, input_data: torch.Tensor, inputs: List[str], sample: bool = True) -> Dict[str, Any]:
        self.model.eval()
        with torch.no_grad():
            out, _ = self.model.generate(input_data, self.helper.args.lm.example_window, sample=sample)
        self.model.train()

        res = {}
        for i, ins in enumerate(inputs):
            outs = self.mark_token(self.train_set.vocabulary.to_string(out[-2*self.helper.args.lm.example_window:, i].cpu().numpy().tolist()), self.helper.args.lm.example_window - 1)
            text = f"Input: \n\n {self.mark_token(inputs[i], self.helper.args.lm.example_window - 1)}\n\nOutput: {outs}"
            res[f"examples/{i}"] = framework.visualize.plot.Text(text)

        return res

