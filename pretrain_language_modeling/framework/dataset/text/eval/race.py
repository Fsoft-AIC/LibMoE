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


class RACE:
    URL = "http://www.cs.cmu.edu/~glai1/data/race/RACE.tar.gz"
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

        self.splits = ["test/high", "test/middle"]  # RACE/RACE/test/high, RACE/RACE/test/middle
        self.data = []

        # with utils.LockFile(self.cache_dir+"lock"):
        self.download()

        self.load_dataset()

        self.maxlen = max(d["max_length"] for d in self.data)

    def __len__(self):
        return len(self.data)

    def download(self):
        if not os.path.exists(self.cache_dir + "RACE"):
            utils.download(self.URL, self.cache_dir)

    def load_dataset(self):
        for si, split in enumerate(self.splits):
            split_path = os.path.join(self.cache_dir, "RACE", split)

            for file in os.listdir(split_path):
                file_path = os.path.join(split_path, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                article = data["article"]
                questions = data["questions"]
                options_lst = data["options"]

                for i, question in enumerate(questions):
                    options = options_lst[i]
                    ctx = self.vocabulary.sentence_to_indices("Article: " + article + "\nQuestion: " + question + "\nAnswer:")
                    endings = [self.vocabulary.sentence_to_indices(" " + e) for e in options]

                    label = data["answers"][i]
                    answer_id = ord(label) - ord("A")

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