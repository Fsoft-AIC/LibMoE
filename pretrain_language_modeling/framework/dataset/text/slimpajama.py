from typing import List, Optional, Dict, Any
import numpy as np

from .chunked_setencepiece_lm_dataset import ChunkedSentencepieceLMDataset


CHUNK_SIZES = {
    "train": {1: 5912, 2: 5911, 3: 5919, 4: 5917, 5: 5933, 6: 5915, 7: 5906, 8: 5921, 9: 5920, 10: 5912},
    "validation": {1: 6279, 2: 6278, 3: 6286, 4: 6284, 5: 6301},
    "test": {1: 6282, 2: 6273, 3: 6289, 4: 6288, 5: 6279}
}

TYPE_MAP = {
    "train": "train",
    "validation": "holdout",
    "test": "holdout"
}


# _DATA_URL = "https://huggingface.co/datasets/cerebras/SlimPajama-627B/resolve/main/{split}/chunk{chunk}/example_{type}_{index}.jsonl.zst"
_DATA_URL = "https://huggingface.co/datasets/cerebras/SlimPajama-627B/resolve/main/{split}/chunk{chunk}/example_{type}_{index}.jsonl.zst"


class SlimPajama(ChunkedSentencepieceLMDataset):
    TOKENIZER_N_FILES = 200

    MAP = {}

    def get_url(self, index: int, split: Optional[str] = None) -> str:
        split = split or self.split

        chunk = self.MAP[split]["chunks"][index]
        index = self.MAP[split]["indices"][index]
        print(_DATA_URL.format(split=split, index=index, chunk=chunk, type=TYPE_MAP[split]))

        return _DATA_URL.format(
            split=split, index=index, chunk=chunk, type=TYPE_MAP[split])

    def get_n_shards(self, split: Optional[str] = None) -> int:
        split = split or self.split
        return len(self.MAP[split]["indices"])

    def __init__(self, unroll_len: int, n_extra: int = 1, split: str = 'train',
                 cache_dir: str = "./cache", n_tokens: int = 8000, token_limit: Optional[int] = None) -> None:

        if not self.MAP:
            print(f"{self.__class__.__name__}: Generating map...")
            for splt in CHUNK_SIZES.keys():
                indices = []
                chunks = []
                for chunk, cnt in CHUNK_SIZES[splt].items():
                    indices += list(range(cnt))
                    chunks += [chunk] * cnt

                rng = np.random.default_rng(123)
                perm = rng.permutation(len(indices)).tolist()

                indices = [indices[i] for i in perm]
                chunks = [chunks[i] for i in perm]

                self.MAP[splt] = {
                    "indices": indices,
                    "chunks": chunks
                }
            print("Map done.")

        super().__init__(unroll_len, n_extra, split, cache_dir, n_tokens, token_limit)
