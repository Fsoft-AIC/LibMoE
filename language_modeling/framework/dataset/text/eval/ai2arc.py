import json
from .probability_compare_dataset import ProbabilityCompareDataset
from framework import data_structures, utils
from typing import Optional
import os
import re
import numpy as np
from .probability_compare_dataset import ProbabilityCompareTest

# Format based on https://github.com/EleutherAI/lm-evaluation-harness/blob/86319a9b14ddae2030bc6e0fdddd47fc7d0bb525/lm_eval/tasks/arc/arc_easy.yaml

class AI2ARC:
    URL = "https://s3-us-west-2.amazonaws.com/ai2-website/data/ARC-V1-Feb2018.zip"

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

        self.splits = ["ARC-Easy", "ARC-Challenge"]
        self.data = []

        # with utils.LockFile(self.cache_dir+"lock"):
        self.download()

        self.load_dataset()

        self.maxlen = max(d["max_length"] for d in self.data)

    def __len__(self):
        return len(self.data)

    def download(self):
        if not os.path.exists(self.cache_dir+"data/ARC-V1-Feb2018-2"):
            print("Downloading AI2ARC dataset...")
            os.makedirs(self.cache_dir+"data/", exist_ok=True)
            utils.download(self.URL, self.cache_dir+"data/", ignore_if_exists=False)
            print("Done.")

    def load_dataset(self):
        for si, split in enumerate(self.splits):
            with open(f"{self.cache_dir}data/ARC-V1-Feb2018-2/{split}/{split}-Test.jsonl", "r") as f:
                for line in f:
                    line = json.loads(line)
                    question = line["question"]["stem"]

                    ctx = self.vocabulary.sentence_to_indices("Question: " + question + "\nAnswer:")

                    endings = [self.vocabulary.sentence_to_indices(" " + e["text"]) for e in line["question"]["choices"]]
                    labels = [e["label"] for e in line["question"]["choices"]]

                    answer_id = labels.index(line["answerKey"])

                    options = [ctx + endings[answer_id]]
                    for i, e in enumerate(endings):
                        if i != answer_id:
                            options.append(ctx + e)


                    if len(options) != 4:
                        print(f"{self.__class__.__name__}: WARNING: Wrong number of options in {split} split: {len(options)}")
                        continue

                    assert len(options) == 4
                    self.data.append({
                        "options": options,
                        "max_length": max(len(i) for i in options),
                        "prefix_length": len(ctx),
                        "group": si
                    })

    def __getitem__(self, idx):
        data = self.data[idx]

        res = {
            "sentence_good": np.array(data["options"][0], dtype=self.dtype),
            "good_len": len(data["options"][0]),
            "prefix_len": data["prefix_length"],
            "max_length": data["max_length"],
            "group": data["group"]
        }

        for i, d in enumerate(data["options"][1:]):
            res[f"sentence_bad_{i}"] = np.array(d, dtype=self.dtype)
            res[f"bad_len_{i}"] = len(d)

        return res

    def start_test(self):
        return ProbabilityCompareTest(self.splits, n_ways=4, normalize_by_length=True)
