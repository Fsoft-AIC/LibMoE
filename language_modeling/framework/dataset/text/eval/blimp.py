import json
import os
from .probability_compare_dataset import ProbabilityCompareDataset, ProbabilityCompareTest


class BLiMP(ProbabilityCompareDataset):
    URL = "https://github.com/alexwarstadt/blimp/raw/master/BLiMP.zip"

    def load_dataset(self):
        for f in os.listdir(f"{self.cache_dir}data/data/"):
            if not f.endswith(".jsonl"):
                continue

            name = os.path.splitext(f)[0]
            self.names.append(name)
            self.data.append([])

            with open(f"{self.cache_dir}data/data/{f}", "r") as f:
                for line in f:
                    line = json.loads(line)
                    self.data[-1].append({
                        "sentence_good": self.vocabulary.sentence_to_indices(self.sep + line["sentence_good"]),
                        "sentence_bad": self.vocabulary.sentence_to_indices(self.sep + line["sentence_bad"]),
                    })

            self.group_offsets.append(len(self.idx_to_group))
            self.idx_to_group += [len(self.names) - 1] * len(self.data[-1])

    def start_test(self):
        return ProbabilityCompareTest(self.names, normalize_by_length=False)
