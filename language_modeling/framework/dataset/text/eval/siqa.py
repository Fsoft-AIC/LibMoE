import os
import json
import pandas as pd
import json
import os
import numpy as np
from typing import List, Optional, Dict
from collections import Counter
import random
from framework import data_structures, utils
from framework.utils.distributed_ops import reduce_any as ra
import torch
import torch.nn.functional as F
import re
import string
import sys
from .probability_compare_dataset import ProbabilityCompareTest


class SIQA:
    URL = "https://storage.googleapis.com/ai2-mosaic/public/socialiqa/socialiqa-train-dev.zip"
    SUPPORTS_DISTRIBUTED = True
    VERSION = "1.0"

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

        self.splits = ["dev"]
        self.data = []

        # with utils.LockFile(self.cache_dir+"lock"):
        self.download()

        self.load_dataset()

        self.maxlen = max(d["max_length"] for d in self.data)

    def __len__(self):
        return len(self.data)

    def download(self):
        if not os.path.exists(self.cache_dir + "socialiqa-train-dev"):
            utils.download(self.URL, self.cache_dir, ignore_if_exists=False)

    def load_dataset(self):
        for si, split in enumerate(self.splits):
            # loading label
            with open(f"{self.cache_dir}socialiqa-train-dev/{split}-labels.lst", "r") as f:
                labels = f.read().splitlines()

            # loading data
            with open(f"{self.cache_dir}socialiqa-train-dev/{split}.jsonl", "r") as f:
                for i, line in enumerate(f):
                    # {"context": "Tracy didn't go home that evening and resisted Riley's attacks.", "question": "What does Tracy need to do before this?", "answerA": "make a new plan", "answerB": "Go home and see Riley", "answerC": "Find somewhere to go"}

                    line = json.loads(line)

                    context = line["context"]
                    question = line["question"]
                    options = [line["answerA"], line["answerB"], line["answerC"]]

                    ctx = self.vocabulary.sentence_to_indices(f"Context: {context}\nQuestion: {question}\nAnswer:")
                    endings = [self.vocabulary.sentence_to_indices(" " + e) for e in options]

                    label = int(labels[i])
                    answer_id = label - 1

                    options = [ctx + endings[answer_id]]
                    for i, e in enumerate(endings):
                        if i != answer_id:
                            options.append(ctx + e)

                    if len(options) != 3:
                        print(f"{self.__class__.__name__}: WARNING: Wrong number of options in {split} split: {len(options)}")
                        continue

                    assert len(options) == 3
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
            "group": data['group']
        }

        for i, d in enumerate(data["options"][1:]):
            res[f"sentence_bad_{i}"] = np.array(d, dtype=self.dtype)
            res[f"bad_len_{i}"] = len(d)

        return res

    def start_test(self):
        return ProbabilityCompareTest(self.splits, n_ways=3, normalize_by_length=True)


