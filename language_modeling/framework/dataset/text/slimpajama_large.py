from typing import Optional
import numpy as np
import os

from ...utils.download import download
from .chunked_setencepiece_lm_dataset import ChunkedSentencepieceLMDataset

pq = None


DATASET_REPO = "https://huggingface.co/datasets/DavidNguyen/LLAVA-LibMoE/tree/main/data/SlimPajama-627B"

# if this repo wrong, use this one: DATASET_REPO = https://huggingface.co/datasets/gmongaras/SlimPajama-627B_Reupload/resolve/main/data
SPLIT_FILE_COUNTS = {
    "train": 250,
    "validation": 30,
    "test": 30,
}


class SlimPajamaLarge(ChunkedSentencepieceLMDataset):
    TOKENIZER_N_FILES = 200

    MAP = {}

    def line_iterator(self, url: str):
        local_dir = os.path.join(self._cache_dir_base, "raw")
        tmp_dir = os.path.join(local_dir, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)

        file_name = url.rsplit("/", 1)[-1]
        target_file = os.path.join(local_dir, file_name)

        if not os.path.exists(target_file):
            tmp_file = os.path.join(tmp_dir, file_name)
            print(f"Downloading {url}")
            download(url, tmp_file, extract=False)
            os.rename(tmp_file, target_file)

        parquet_file = pq.ParquetFile(target_file)
        for batch in parquet_file.iter_batches(columns=["text"], batch_size=8192):
            for text in batch.column("text").to_pylist():
                yield text + "<STORY_SEP>" if text else text

    def get_url(self, index: int, split: Optional[str] = None) -> str:
        split = split or self.split
        shard_index = self.MAP[split][index]
        shard_count = SPLIT_FILE_COUNTS[split]
        return f"{DATASET_REPO}/{split}-{shard_index:05d}-of-{shard_count:05d}.parquet"

    def get_n_shards(self, split: Optional[str] = None) -> int:
        split = split or self.split
        return len(self.MAP[split])

    def __init__(self, unroll_len: int, n_extra: int = 1, split: str = 'train',
                 cache_dir: str = "./cache", n_tokens: int = 8000,
                 token_limit: Optional[int] = None) -> None:
        global pq
        if pq is None:
            import pyarrow.parquet as pq

        if not self.MAP:
            print(f"{self.__class__.__name__}: Generating map...")
            rng = np.random.default_rng(123)
            for split_name, shard_count in SPLIT_FILE_COUNTS.items():
                self.MAP[split_name] = rng.permutation(shard_count).tolist()
            print("Map done.")

        super().__init__(unroll_len, n_extra, split, cache_dir, n_tokens, token_limit)
