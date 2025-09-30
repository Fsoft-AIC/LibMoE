import json
from typing import Optional, Dict, Any, List
import torch
import os
import numpy as np
from framework.utils.distributed_ops import reduce_any as ra
import torch.nn.functional as F
from framework import data_structures, utils


class ProbabilityCompareTest:
    SUPPORTS_DISTRIBUTED = True

    def __init__(self, group_names: List[str], batch_dim: int = 1, n_ways: int = 2, normalize_by_length = True):
        if batch_dim != 1:
            raise NotImplementedError("Batch dim must be 1")

        self.group_names = group_names
        self.n_ways = n_ways
        self.normalize_by_length = normalize_by_length

        self.counters = [
            {
                "n_ok": 0,
                "n_total": 0,
            } for _ in group_names
        ]

    @torch.no_grad()
    def get_lprobs(self, logits: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor,
                   prefix_length: Optional[torch.Tensor] = None):
        targets = targets[1:]
        lprobs = F.cross_entropy(logits.flatten(end_dim=-2), targets.flatten().long(), reduction="none").view_as(targets).double()
        lprobs.masked_fill_(torch.arange(lprobs.shape[0], device=lprobs.device).unsqueeze(1) >= (lengths.unsqueeze(0) - 1), 0)
        if self.normalize_by_length:
            if prefix_length is None:
                raise ValueError("prefix_length must be provided if normalize_by_length is True")

            lprobs.masked_fill_(torch.arange(lprobs.shape[0], device=lprobs.device).unsqueeze(1) < (prefix_length.unsqueeze(0) - 1), 0)
            lprobs = -lprobs.sum(0)
            assert (lengths > prefix_length).all().item()
            lprobs /= (lengths - prefix_length).double()
        else:
            lprobs = -lprobs.sum(0)
        return lprobs

    @torch.no_grad()
    def step(self, good_lprob: torch.Tensor, bad_lprobs: List[torch.Tensor], data: Dict[str, torch.Tensor]):
        ok = torch.ones_like(good_lprob, dtype=torch.bool)

        for bad_lprob in bad_lprobs:
            ok &= good_lprob > bad_lprob

        for i in data["group"].int().unique().cpu().numpy().tolist():
            i = int(i)
            mask = data["group"] == i
            n_ok = ok[mask].sum()
            n_total = mask.sum()
            self.counters[i]["n_ok"] += int(n_ok.item())
            self.counters[i]["n_total"] += int(n_total)

    @property
    def accuracy(self):
        return ra(sum(c["n_ok"] for c in self.counters)) / ra(sum(c["n_total"] for c in self.counters))

    def plot(self) -> Dict[str, Any]:
        res = {}
        print("Counter zero test")
        for i, name in enumerate(self.group_names):
            c = ra(self.counters[i]["n_total"])
            if c == 0:
                print(f"  Counter 0 for {i}: {name}")

        for i, name in enumerate(self.group_names):
            res[f"accuracy/{name}"] = ra(self.counters[i]["n_ok"]) / ra(self.counters[i]["n_total"])

        res["accuracy/group_average"] = sum(res.values()) / len(res)
        res["accuracy/seq_average"] = self.accuracy
        return res


class ProbabilityCompareDataset:
    URL = None

    def load_dataset(self):
        raise NotImplementedError

    def download(self):
        if not os.path.exists(self.cache_dir+"data/data"):
            utils.download(self.URL, self.cache_dir+"data", ignore_if_exists=True)

    def __init__(self, vocabulary: data_structures.vocabulary.Vocabulary,
                 cache_dir: str = "./cache", sep: Optional[str] = None) -> None:
        self.vocabulary = vocabulary
        self.sep = sep if sep is not None else ""

        if len(self.vocabulary) <= 256:
            self.dtype = np.uint8
        if len(self.vocabulary) < 32768:
            self.dtype = np.int16
        else:
            self.dtype = np.int32

        self.cache_dir = f"{cache_dir}/{self.__class__.__name__}/"
        os.makedirs(self.cache_dir, exist_ok=True)

        # with utils.LockFile(self.cache_dir+"lock"):
        #     self.download()

        self.data = []
        self.names = []
        self.idx_to_group = []
        self.group_offsets = []

        print(f"Loading {self.__class__.__name__}")
        self.load_dataset()

        for d in self.data:
            assert len(d) > 0

        print("Done...")
        self.n_inputs = sum(len(v) for v in self.data)
        self.maxlen = max(max(max(len(i["sentence_good"]), len(i["sentence_bad"])) for i in v) for v in self.data)

    def __len__(self):
        return self.n_inputs

    def __getitem__(self, idx):
        group = self.idx_to_group[idx]
        idx = idx - self.group_offsets[group]
        data = self.data[group][idx]

        return {
            "sentence_good": np.array(data["sentence_good"], dtype=self.dtype),
            "sentence_bad_0": np.array(data["sentence_bad"], dtype=self.dtype),
            "good_len": len(data["sentence_good"]),
            "bad_len_0": len(data["sentence_bad"]),
            "max_length": max(len(data["sentence_good"]), len(data["sentence_bad"])),
            "group": group
        }

    def start_test(self):
        return ProbabilityCompareTest(self.names)
