import json
from .probability_compare_dataset import ProbabilityCompareDataset
from framework import data_structures, utils
from typing import Optional
import os
import re
import numpy as np
from .probability_compare_dataset import ProbabilityCompareTest

# preprocessing based on https://github.com/EleutherAI/lm-evaluation-harness/blob/86319a9b14ddae2030bc6e0fdddd47fc7d0bb525/lm_eval/tasks/hellaswag/utils.py

class HellaSwag:
    URL = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
    CLEANUP_REGEX = re.compile(r"\\[.*?\\]")

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
        self.download()

        self.load_dataset()

        self.maxlen = max(d["max_length"] for d in self.data)

    def __len__(self):
        return len(self.data)

    def download(self):
        if not os.path.exists(self.cache_dir+"data/hellaswag_val.json"):
            os.makedirs(self.cache_dir+"data/", exist_ok=True)
            # utils.download(self.URL, self.cache_dir+"data/", ignore_if_exists=True)

    def preprocess(self, text):
        text = text.strip()
        # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
        text = text.replace(" [title]", ". ")
        text = self.CLEANUP_REGEX.sub("", text)
        text = text.replace("  ", " ")
        return text

    def load_dataset(self):
        target = "hellaswag_val.jsonl"

        with open(f"{self.cache_dir}data/{target}", "r") as f:
            for line in f:
                line = json.loads(line)

                ctx = self.preprocess(line["activity_label"] + ": " + line["ctx_a"] + " " + line["ctx_b"].capitalize())
                ctx = self.vocabulary.sentence_to_indices(ctx)

                endings = [self.vocabulary.sentence_to_indices(" " + self.preprocess(e)) for e in line["endings"]]
                options = [ctx + endings[line["label"]]]
                for i, e in enumerate(endings):
                    if i != line["label"]:
                        options.append(ctx + e)

                assert len(options) == 4
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
        return ProbabilityCompareTest(["val"], n_ways=4, normalize_by_length=True)
