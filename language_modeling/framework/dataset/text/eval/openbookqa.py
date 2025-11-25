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

class OpenBookQA:
    URL = "https://s3-us-west-2.amazonaws.com/ai2-website/data/OpenBookQA-V1-Sep2018.zip"
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

        self.splits = ["test"]
        self.data = []

        # with utils.LockFile(self.cache_dir+"lock"):
        self.download()

        self.load_dataset()

        self.maxlen = max(d["max_length"] for d in self.data)

    def __len__(self):
        return len(self.data)

    def download(self):
        if not os.path.exists(self.cache_dir + "OpenBookQA-V1-Sep2018.zip"):
            os.makedirs(self.cache_dir, exist_ok=True)
            utils.download(self.URL, self.cache_dir, ignore_if_exists=False)

    def load_dataset(self):
        for si, split in enumerate(self.splits):
            with open(f"{self.cache_dir}OpenBookQA-V1-Sep2018/Data/Main/{split}.jsonl", "r") as f:
                for line in f:
                    line = json.loads(line)

                    # {"id": "8-343", "question": {"stem": "A person wants to start saving money so that they can afford a nice vacation at the end of the year. After looking over their budget and expenses, they decide the best way to save money is to", "choices": [{"text": "make more phone calls", "label": "A"}, {"text": "quit eating lunch out", "label": "B"}, {"text": "buy less with monopoly money", "label": "C"}, {"text": "have lunch with friends", "label": "D"}]}, "fact1": "using less resources usually causes money to be saved", "humanScore": "1.00", "clarity": "2.00", "turkIdAnonymized": "b356d338b7", "answerKey": "B"}

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
