import numpy as np
import os
from typing import List, Optional, Dict
from .probability_compare_dataset import ProbabilityCompareTest
from framework import data_structures, utils


class ChildrenBooksTest:
    URL = "https://huggingface.co/datasets/cbt/resolve/7503a0643517afe02a86e4750d375a9686008efa/data/CBTest.tgz"

    def detokenize(self, text):
        # from https://github.com/declare-lab/instruct-eval/blob/main/lm_eval/tasks/cbt.py
        text = text.replace(" '", "'")
        text = text.replace(" \n", "\n")
        text = text.replace("\n ", "\n")
        text = text.replace(" n't", "n't")
        text = text.replace("`` ", '"')
        text = text.replace("''", '"')
        # punctuation
        text = text.replace(" :", ":")
        text = text.replace(" ;", ";")
        text = text.replace(" !", "!")
        text = text.replace(" ?", "?")
        text = text.replace(" ,", ",")
        text = text.replace(" .", ".")
        return text

    def process_lines(self, lines: List[str]) -> Dict[str, List[List[int]]]:
        question = " ".join(self.detokenize(l) for l in lines[:-1])
        answer = [l for l in lines[-1].split("\t") if l]
        aline = self.detokenize(answer[0])
        answer_ok = answer[1]
        answer_options = [a for a in answer[2].split("|") if a]

        aline = aline.split("XXXXX")
        question = question + " " + aline[0]
        postfix_good = answer_ok + aline[1]

        postfix_bad = []
        for a in answer_options:
            if a != answer_ok:
                postfix_bad.append(a + aline[1])

        question = self.vocabulary.sentence_to_indices(question)
        res = [question + self.vocabulary.sentence_to_indices(postfix_good)]
        for a in postfix_bad:
            res.append(question + self.vocabulary.sentence_to_indices(a))

        return {
            "options": res,
            "prefix_length": len(question),
            "max_length": max(len(i) for i in res)
        }

    def add_last_lines(self, lines: List[str]):
        this_q = self.process_lines(lines)
        if self.length_limit is not None:
            for s in this_q:
                if len(s) > self.length_limit:
                    self.length_limit_skipped += 1
                    return

        if len(this_q["options"]) != 10:
            opt = lines[-1].split("\t")[-1]
            print(f"   WARNING: {self.__class__.__name__}: Invalid number of options: {opt}")
            return
        self.data[-1].append(this_q)

    def __init__(self, vocabulary: data_structures.vocabulary.Vocabulary,
                 cache_dir: str = "./cache", length_limit: Optional[int] = None):

        self.vocabulary = vocabulary
        self.length_limit = length_limit
        self.length_limit_skipped = 0

        if len(self.vocabulary) <= 256:
            self.dtype = np.uint8
        if len(self.vocabulary) < 32768:
            self.dtype = np.int16
        else:
            self.dtype = np.int32

        self.cache_dir = f"{cache_dir}/{self.__class__.__name__}/"
        os.makedirs(self.cache_dir, exist_ok=True)

        # with utils.LockFile(self.cache_dir+"lock"):
        if not os.path.exists(self.cache_dir+"data/CBTest"):
            utils.download(self.URL, self.cache_dir+"data", ignore_if_exists=True)

        self.data_dir = f"{self.cache_dir}/data/CBTest/data"

        self.data = []
        self.names = []
        self.idx_to_group = []
        self.group_offsets = []

        print(f"Loading {self.__class__.__name__}...")
        for fn in os.listdir(self.data_dir):
            if "_test_" not in fn:
                continue

            self.data.append([])
            with open(f"{self.data_dir}/{fn}", "r") as f:
                lines = []
                for l in f:
                    l = l.strip()
                    if l:
                        lines.append(l[l.index(" ")+1:])
                    else:
                        self.add_last_lines(lines)
                        lines.clear()

            if lines:
                self.add_last_lines(lines)

            self.names.append(fn.split("_")[1])
            self.group_offsets.append(len(self.idx_to_group))
            self.idx_to_group += [len(self.names) - 1] * len(self.data[-1])

        print("Done...")
        self.n_inputs = sum(len(v) for v in self.data)
        self.maxlen = max(max(max(len(i) for i in q["options"]) for q in splits) for splits in self.data)
        # self.maxoptions = max(max(len(q) for q in splits) for splits in self.data)

        if self.length_limit is not None:
            print(f"{self.__class__.__name__}: {self.length_limit_skipped} stories removed because of the length limit ({self.length_limit_skipped*100/(self.length_limit_skipped + self.n_inputs):.2f}%)")
        print(f"{self.__class__.__name__}: Loaded {self.n_inputs} sequences.")

    def __len__(self):
        return self.n_inputs

    def __getitem__(self, idx):
        group = self.idx_to_group[idx]
        idx = idx - self.group_offsets[group]
        data = self.data[group][idx]

        res = {
            "sentence_good": np.array(data["options"][0], dtype=self.dtype),
            "good_len": len(data["options"][0]),
            "prefix_len": data["prefix_length"],
            "max_length": data["max_length"],
            "group": group
        }

        for i, d in enumerate(data["options"][1:]):
            res[f"sentence_bad_{i}"] = np.array(d, dtype=self.dtype)
            res[f"bad_len_{i}"] = len(d)

        return res

    def start_test(self):
        return ProbabilityCompareTest(self.names, n_ways=10, normalize_by_length=False)