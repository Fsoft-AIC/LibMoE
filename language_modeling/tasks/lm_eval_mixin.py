import framework
from framework.dataset import Lambada, BLiMP, ChildrenBooksTest, HellaSwag, PIQA, AI2ARC, \
    MMLU, CommonsenseQA, OpenBookQA, RACE, SIQA, Winogrande
from typing import Tuple, Any
from tqdm import tqdm
import torch
from framework.loader.dataset_splitter import DatasetSplitter
import torch.utils.data
import torch.nn.functional as F

from framework.task import args


@args
def a(parser: framework.helpers.ArgumentParser):
    parser.add_argument("-lm.eval.blimp.batch_mul", default=16)
    parser.add_argument("-lm.eval.enabled", default=True)
    parser.add_argument("-lm.eval.lambada.enabled", default=False)
    parser.add_argument("-lm.eval.cbt.batch_mul", default=1)
    parser.add_argument("-lm.eval.cbt.length_limit", default="none", parser=parser.int_or_none_parser)
    parser.add_argument("-lm.eval.cbt.enabled", default=False)
    parser.add_argument("-lm.eval.cbt.end_only", default=False)
    parser.add_argument("-lm.eval.blimp.enabled", default=False)
    parser.add_argument("-lm.eval.hellaswag.enabled", default=False)
    parser.add_argument("-lm.eval.hellaswag.batch_mul", default=16)
    parser.add_argument("-lm.eval.piqa.enabled", default=False)
    parser.add_argument("-lm.eval.piqa.batch_mul", default=16)
    parser.add_argument("-lm.eval.ai2arc.enabled", default=False)
    parser.add_argument("-lm.eval.ai2arc.batch_mul", default=4)

    # additional datasets
    parser.add_argument("-lm.eval.mmlu.enabled", default=False)
    parser.add_argument("-lm.eval.openbookqa.enabled", default=False)
    parser.add_argument("-lm.eval.race.enabled", default=False)
    parser.add_argument("-lm.eval.siqa.enabled", default=False)
    parser.add_argument("-lm.eval.winogrande.enabled", default=False)
    parser.add_argument("-lm.eval.commonsenseqa.enabled", default=False)


class LMEvalMixin:
    helper: framework.helpers.TrainingHelper

    def create_datasets(self):
        self.state_enabled = True
        self.pad_quantum = None
        # sep = "<STORY_SEP>"
        sep = ""
        self.prob_compare_valid_sets = framework.data_structures.DotDict()

        if self.helper.args.lm.eval.enabled:
            if self.helper.args.lm.eval.lambada.enabled:
                self.valid_sets.lambada = Lambada(self.train_set.vocabulary, sep=sep)
            if self.helper.args.lm.eval.blimp.enabled:
                self.prob_compare_valid_sets.blimp = BLiMP(self.train_set.vocabulary, sep=sep)
            if self.helper.args.lm.eval.cbt.enabled:
                self.prob_compare_valid_sets.cbt = ChildrenBooksTest(self.train_set.vocabulary, length_limit=self.helper.args.lm.eval.cbt.length_limit)
            if self.helper.args.lm.eval.hellaswag.enabled:
                self.prob_compare_valid_sets.hellaswag = HellaSwag(self.train_set.vocabulary)
            if self.helper.args.lm.eval.piqa.enabled:
                self.prob_compare_valid_sets.piqa = PIQA(self.train_set.vocabulary)
            if self.helper.args.lm.eval.ai2arc.enabled:
                self.prob_compare_valid_sets.ai2arc = AI2ARC(self.train_set.vocabulary)
            # additional datasets
            if self.helper.args.lm.eval.mmlu.enabled:
                self.prob_compare_valid_sets.mmlu = MMLU(self.train_set.vocabulary)
            if self.helper.args.lm.eval.openbookqa.enabled:
                self.prob_compare_valid_sets.openbookqa = OpenBookQA(self.train_set.vocabulary)
            if self.helper.args.lm.eval.race.enabled:
                self.prob_compare_valid_sets.race = RACE(self.train_set.vocabulary)
            if self.helper.args.lm.eval.siqa.enabled:
                self.prob_compare_valid_sets.siqa = SIQA(self.train_set.vocabulary)
            if self.helper.args.lm.eval.winogrande.enabled:
                self.prob_compare_valid_sets.winogrande = Winogrande(self.train_set.vocabulary)
            if self.helper.args.lm.eval.commonsenseqa.enabled:
                self.prob_compare_valid_sets.commonsenseqa = CommonsenseQA(self.train_set.vocabulary)


    def run_model(self, data, *args, **kwargs):
        if (not self.state_enabled) and hasattr(self.model_interface, "reset_state"):
            self.model_interface.reset_state()

        if self.pad_quantum is not None:
            olen = data["data"].shape[0]
            # Quantize length so that we avoid continous recompilation.
            tlen = ((olen + self.pad_quantum - 1) // self.pad_quantum) * self.pad_quantum
            if tlen != olen:
                data = {k: v for k, v in data.items()}
                data["data"] = F.pad(data["data"], [0] * ((data['data'].ndim-1)*2) + [0, tlen-olen], value=0, mode="constant")

        res, d  = super().run_model(data, *args, **kwargs)

        if self.pad_quantum is not None and tlen != olen:
            res.outputs = res.outputs[:olen-1]

        return res, d

    def get_test_logits(self, data: torch.Tensor) -> Tuple[Any, Any]:
        self.model_interface.reset_state()
        olen = data.shape[0]
        res, _ = self.run_model({"data": data})
        res = self.model_interface.decode_outputs(res)
        return res[:olen-1]

    @torch.no_grad()
    def validate_on_pc_dataset(self, name) -> Tuple[Any, float]:
        self.model.eval()

        ds = self.prob_compare_valid_sets[name]
        print(f"Validating on {ds.__class__.__name__}")
        state = self.model_interface.state

        self.pad_quantum = 128
        if 2*self.pad_quantum > self.prob_compare_valid_sets[name].maxlen:
            self.pad_quantum = self.prob_compare_valid_sets[name].maxlen

        test = ds.start_test()
        for d in tqdm(self.valid_loaders[name]):
            d = self.prepare_data(d)
            prefix_len = d.get("prefix_len")
            good_lprob = test.get_lprobs(self.get_test_logits(d["sentence_good"]), d["sentence_good"], d["good_len"], prefix_len)
            bad_lprobs = []
            for i in range(test.n_ways-1):
                bad_lprobs.append(test.get_lprobs(self.get_test_logits(d[f"sentence_bad_{i}"]), d[f"sentence_bad_{i}"], d[f"bad_len_{i}"], prefix_len))

            test.step(good_lprob, bad_lprobs, d)

        self.model_pad_quantum = None
        self.model_interface.state = state
        self.model.train()
        print(f"{ds.__class__.__name__} accuracy: {test.accuracy}")
        return {f"{name}/{k}": v for k, v in test.plot().items()}

    def validate_on_name(self, name: str) -> Tuple[Any, float]:
        print(f"Starting validation on {name}...")
        self.state_enabled = name not in {"lambada"}
        self.pad_quantum = 128 if name == "lambada" else None
        res = self.validate_on(self.valid_sets[name], self.valid_loaders[name])
        self.pad_quantum = None
        self.state_enabled = True
        return res

    def create_loaders(self):
        super().create_loaders()
        for name, ds in self.prob_compare_valid_sets.items():
            batch_size = self.helper.args.lm.eval.get(name, {}).get("batch_mul", 1) * self.test_batch_size

            ds = DatasetSplitter(ds, n_partitions=self.helper.dist_env.world_size,
                                 current=self.helper.dist_env.rank)

            kwargs = {}
            if "max_length" in ds[0]:
                kwargs["batch_sampler"] = framework.loader.sampler.BucketedSampler(
                    ds, batch_size, infinite=False, long_first=True, random_order=False, seed=0,
                    length_key_names=["max_length"])
            else:
                kwargs["batch_size"] = batch_size

            self.valid_loaders.update({
                name: torch.utils.data.DataLoader(ds,
                    collate_fn=framework.loader.collate.VarLengthCollate(batch_dim=self.batch_dim),
                    num_workers=self.VALID_NUM_WORKERS,
                    **kwargs)
            })


    def validate(self) -> Tuple[Any, float]:
        is_end = self.helper.state.iter == self.helper.args.stop_after

        res = super().validate()
        for name in self.prob_compare_valid_sets:
            print(f"Starting validation on {name}...")
            if is_end or (not self.helper.args.lm.eval.get(name, {}).get("end_only", False)) or name == "cbt":
                res.update(self.validate_on_pc_dataset(name))
        return res