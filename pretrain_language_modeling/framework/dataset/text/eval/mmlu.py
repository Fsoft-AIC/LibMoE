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


class MMLU:
    """MMLU dataset implementation following framework patterns"""

    URL = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"
    SUPPORTS_DISTRIBUTED = True
    VERSION = "1.0"

    # List of all MMLU sub-tasks
    TASKS = [
        'abstract_algebra',
        'anatomy',
        'astronomy',
        'business_ethics',
        'clinical_knowledge',
        'college_biology',
        'college_chemistry',
        'college_computer_science',
        'college_mathematics',
        'college_medicine',
        'college_physics',
        'computer_security',
        'conceptual_physics',
        'econometrics',
        'electrical_engineering',
        'elementary_mathematics',
        'formal_logic',
        'global_facts',
        'high_school_biology',
        'high_school_chemistry',
        'high_school_computer_science',
        'high_school_european_history',
        'high_school_geography',
        'high_school_government_and_politics',
        'high_school_macroeconomics',
        'high_school_mathematics',
        'high_school_microeconomics',
        'high_school_physics',
        'high_school_psychology',
        'high_school_statistics',
        'high_school_us_history',
        'high_school_world_history',
        'human_aging',
        'human_sexuality',
        'international_law',
        'jurisprudence',
        'logical_fallacies',
        'machine_learning',
        'management',
        'marketing',
        'medical_genetics',
        'miscellaneous',
        'moral_disputes',
        'moral_scenarios',
        'nutrition',
        'philosophy',
        'prehistory',
        'professional_accounting',
        'professional_law',
        'professional_medicine',
        'professional_psychology',
        'public_relations',
        'security_studies',
        'sociology',
        'us_foreign_policy',
        'virology',
        'world_religions'
    ]

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

        self.splits = ["MMLU"]
        self.data = []

        # with utils.LockFile(self.cache_dir+"lock"):
        self._download_and_prepare()

        self._load_and_process_data()

        self.maxlen = max(d["max_length"] for d in self.data)

    def __len__(self):
        return len(self.data)

    def _download_and_prepare(self):
        """Download and prepare the dataset files"""
        data_file = os.path.join(self.cache_dir, "data.tar")

        if not os.path.exists(data_file):
            utils.download(self.URL, data_file, ignore_if_exists=True)
            # Extract tar file
            os.system(f"tar -xf {data_file} -C {self.cache_dir}")
            os.remove(data_file)

    def _get_label_id(self, label: str) -> int:
        # get the label id from the label ["A", "B", "C", "D"], A is 0, B is 1, C is 2, D is 3
        return ord(label) - ord("A")

    def _load_and_process_data(self):
        for task in self.TASKS:
            for split in ["test"]:
                file_path = os.path.join(self.cache_dir, f"{split}/{task}_{split}.csv")
                if not os.path.exists(file_path):
                    print(f"Skipping {task} {split} because file does not exist")
                    continue

                df = pd.read_csv(file_path)
                for i in range(df.shape[0]):
                    line = list(df.iloc[i])
                    question = str(line[0])
                    options = [str(line[j]) for j in range(1, 5)]
                    label = str(line[5])
                    label_id = self._get_label_id(label)

                    # create ctx
                    ctx = self.vocabulary.sentence_to_indices(f"Question: {question}\nOptions: {options}\nAnswer:")

                    # good option is the option with the label id, bad options are the rest
                    good_option = options[label_id]
                    bad_options = [options[j] for j in range(len(options)) if j != label_id]

                    options = [ctx + self.vocabulary.sentence_to_indices(good_option)] + [ctx + self.vocabulary.sentence_to_indices(bad_option) for bad_option in bad_options]

                    # endings = [ctx + self.vocabulary.sentence_to_indices(" " + option) for option in options]

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
        return ProbabilityCompareTest(self.splits, n_ways=4, normalize_by_length=True)

















