import json
from .probability_compare_dataset import ProbabilityCompareDataset
from framework import data_structures, utils
from typing import Optional
import os
import re
import numpy as np
from .probability_compare_dataset import ProbabilityCompareTest

# preprocessing based on https://github.com/EleutherAI/lm-evaluation-harness/blob/86319a9b14ddae2030bc6e0fdddd47fc7d0bb525/lm_eval/tasks/piqa/piqa.yaml

class PIQA:
    URL = "https://raw.githubusercontent.com/ybisk/ybisk.github.io/master/piqa/data/valid.jsonl"
    URL_LABELS = "https://raw.githubusercontent.com/ybisk/ybisk.github.io/master/piqa/data/valid-labels.lst"

    def __init__(self, vocabulary: data_structures.vocabulary.Vocabulary, cache_dir: str = "./cache") -> None:
        self.cache_dir = f"{cache_dir}/{self.__class__.__name__}/"
        os.makedirs(self.cache_dir, exist_ok=True)

        self.vocabulary = vocabulary
        if len(self.vocabulary) <= 256:
            self.dtype = np.uint8
        if len(self.vocabulary) < 32768:
            self.dtype = np.int16
        else:
            self.dtype = np.int32

        self.data = []

        # with utils.LockFile(self.cache_dir+"lock"):
            # self.download()

        self.load_dataset()

        self.maxlen = max(d["max_length"] for d in self.data)

    def __len__(self):
        return len(self.data)

    def download(self):
        if not os.path.exists(self.cache_dir+"data/valid.json"):
            os.makedirs(self.cache_dir+"data/", exist_ok=True)
            utils.download(self.URL_LABELS, self.cache_dir+"data/", ignore_if_exists=True)
            utils.download(self.URL, self.cache_dir+"data/", ignore_if_exists=True)

    def load_dataset(self):
        with open(f"{self.cache_dir}data/valid-labels.lst", "r") as f:
            labels = f.read().splitlines()

        with open(f"{self.cache_dir}data/valid.jsonl", "r") as f:
            for i, line in enumerate(f):
                line = json.loads(line)
                label = int(labels[i])
                assert label in {0, 1}

                ctx = f"Question: {line['goal'].strip()}\nAnswer:"
                ctx = self.vocabulary.sentence_to_indices(ctx)

                good = f"sol{1 + label}"
                bad = f"sol{2 - label}"

                endings = [" " + line[good], " " + line[bad]]

                options = [ctx + self.vocabulary.sentence_to_indices(e) for e in endings]

                self.data.append({
                    "options": options,
                    "max_length": max(len(i) for i in options),
                    "prefix_length": len(ctx)
                })

    def __getitem__(self, idx):
        data = self.data[idx]

        res = {
            "sentence_good": np.array(data["options"][0], dtype=self.dtype),
            "good_len": len(data["options"][0]),
            "prefix_len": data["prefix_length"],
            "max_length": data["max_length"],
            "group": 0
        }

        for i, d in enumerate(data["options"][1:]):
            res[f"sentence_bad_{i}"] = np.array(d, dtype=self.dtype)
            res[f"bad_len_{i}"] = len(d)

        return res

    def start_test(self):
        return ProbabilityCompareTest(["val"], n_ways=2, normalize_by_length=True)
