
from typing import List, Optional, Dict, Any
import numpy as np
import os
from ...utils.download import download
pd = None

from .chunked_setencepiece_lm_dataset import ChunkedSentencepieceLMDataset

CNT_PER_LANG = {
    "python": 206,
    "html": 802,
    "c++": 214,
    "rust": 40,
    "javascript": 499,
    "scala": 17,
    "haskell": 7,
    "assembly": 3
}

class TheStack(ChunkedSentencepieceLMDataset):
    TOKENIZER_N_FILES = 20

    MAP = {}

    def _get_variant_id(self) -> str:
        lang = "_".join(self.languages)
        return f"{self.__class__.__name__}-{lang}-{self.n_tokens}"

    def line_iterator(self, url: str):
        local_dir = os.path.join(self._cache_dir_base, "raw")
        tmp_dir = os.path.join(local_dir, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)

        lp = url[url.index("main/data/") + len("main/data/"):]
        lp = lp.replace("/", "_")
        target_file = os.path.join(local_dir, lp)

        print("Target file", target_file)
        if not os.path.exists(target_file):
            if self.hf_token is None:
                raise ValueError(f"{self.__class__.__name__} requires HF_TOKEN to be set")

            tmp_file = os.path.join(tmp_dir, lp)
            print(f"Downloading {url}")
            download(url, tmp_file, headers={"Authorization": f"Bearer {self.hf_token}"})
            os.rename(tmp_file, target_file)

        df = pd.read_parquet(target_file, engine='fastparquet')
        for _, j in df.iterrows():
            yield j["content"]

    def get_url(self, index: int, split: Optional[str] = None) -> str:
        split = split or self.split
        lang, index = self.ids[split][index]
        cnt = CNT_PER_LANG[lang]
        return f"https://huggingface.co/datasets/bigcode/the-stack/resolve/main/data/{lang}/train-{index:05d}-of-{cnt:05d}.parquet"

    def get_n_shards(self, split: Optional[str] = None) -> int:
        split = split or self.split
        return len(self.ids[split])

    def mix_ids(self, ids):
        order = list(sorted(ids.keys()))
        res = []
        i = 0
        last_len = -1
        while last_len != len(res):
            last_len = len(res)

            for o in order:
                if i < len(ids[o]):
                    res.append((o, ids[o][i]))
            i += 1
        return res

    def get_min_n_chunks(self) -> int:
        return len(self.languages)

    def get_tokenizer_n_files(self) -> int:
        return max(super().get_tokenizer_n_files(), self.get_min_n_chunks())

    def __init__(self, languages: str, unroll_len: int, n_extra: int = 1, split: str = 'train',
                 cache_dir: str = "./cache/", n_tokens: int = 8000, token_limit: Optional[int] = None) -> None:

        global pd
        if pd is None:
            import pandas as pd

        self.hf_token = os.getenv("HF_TOKEN")
        rng = np.random.default_rng(123)

        self.languages = [l.strip() for l in languages.split(",")]
        self.languages = [l for l in self.languages if l]
        self.languages = list(sorted(self.languages))

        if not self.languages:
            raise ValueError("No languages specified.")

        lang_ids = {
            lang: rng.permutation(CNT_PER_LANG[lang]).tolist() for lang in self.languages
        }

        self.valid_size = {
            lang: max(int(CNT_PER_LANG[lang]*0.1), 1) for lang in self.languages
        }

        valid_ids = {l: lang_ids[l][:self.valid_size[l]] for l in self.languages}
        train_ids = {l: lang_ids[l][self.valid_size[l]:] for l in self.languages}

        self.ids = {
            "train": self.mix_ids(train_ids),
            "validation": self.mix_ids(valid_ids)
        }

        super().__init__(unroll_len, n_extra, split, cache_dir, n_tokens, token_limit, shuffle=True)
