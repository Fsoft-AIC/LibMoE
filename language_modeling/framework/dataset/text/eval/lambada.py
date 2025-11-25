import json
from typing import Optional, Dict, Any
import torch
import os
import numpy as np
from framework.utils.distributed_ops import reduce_any
import torch.nn.functional as F
from framework import data_structures, utils


class LambadaTest:
    SUPPORTS_DISTRIBUTED = True

    def __init__(self, data, vocabulary: data_structures.vocabulary.Vocabulary, batch_dim: int = 1):
        self.batch_dim = batch_dim
        self.time_dim = 1 - self.batch_dim
        self.vocabulary = vocabulary
        # This copy of data is to avoid detokenizing every single time.
        self.data = data
        if self.batch_dim != 1:
            raise NotImplementedError("Batch dim must be 1")

        self.n_ok = 0
        self.n_tok_ok = 0
        self.n_total = 0
        self.loss_total = 0
        self.lm_loss = 0
        self.n_total_tokens = 0

    def step(self, net_out_logits: torch.Tensor, data: Dict[str, torch.Tensor]):
        last_words = [self.data[int(i)]["text"].split(" ")[-1] for i in data["index"].cpu().numpy().tolist()]
        net_out = net_out_logits.argmax(-1)
        for i in range(net_out.shape[self.batch_dim]):
            in_l = int(data["in_len"][i])
            wlen = len(last_words[i])

            out_seq = net_out[in_l - 1 - wlen: in_l - 1, i].cpu().numpy().tolist()
            out_seq = [int(i) for i in out_seq]
            detok = self.vocabulary.to_string(out_seq)
            last_predicted = detok.split(" ")[-1]

            last_word = self.vocabulary.sentence_to_indices(" "+last_words[i])
            last_tok = last_word[-1]
            self.n_tok_ok += int(last_tok == net_out[in_l - 2, i])

            out_end = net_out_logits[in_l - 1 - len(last_word):in_l - 1, i]
            loss = F.cross_entropy(out_end, torch.tensor(last_word, device=out_end.device, dtype=torch.long))

            self.loss_total += loss.cpu().item()

            self.n_ok += int(last_predicted == last_words[i])
            self.n_total += 1

        target = data["data"][1:].contiguous()
        target = target.masked_fill(torch.arange(target.size(0), device=target.device)[:, None] >= (data["in_len"][None] - 1), -100)
        self.lm_loss += F.cross_entropy(net_out_logits.flatten(end_dim=-2), target.flatten().long(), ignore_index=-100, reduction="sum").item()
        self.n_total_tokens += (data["in_len"] - 1).sum().item()


    @property
    def accuracy(self):
        return reduce_any(self.n_ok) / reduce_any(self.n_total)

    def plot(self) -> Dict[str, Any]:
        lm_loss = reduce_any(self.lm_loss) / reduce_any(self.n_total_tokens)

        return {
            "accuracy/total": self.accuracy,
            "accuracy/openai_last_token": reduce_any(self.n_tok_ok) / reduce_any(self.n_total),
            "perplexity": np.exp(self.loss_total / self.n_total),
            "lm_loss": lm_loss,
            "lm_perplexity": np.exp(lm_loss)
        }


class Lambada:
    URL = "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"

    def __init__(self, vocabulary: data_structures.vocabulary.Vocabulary,
                 cache_dir: str = "./cache", sep: Optional[str] = None) -> None:
        self.vocabulary = vocabulary
        self.sep = sep

        if len(self.vocabulary) <= 256:
            self.dtype = np.uint8
        if len(self.vocabulary) < 32768:
            self.dtype = np.int16
        else:
            self.dtype = np.int32

        self.cache_dir = f"{cache_dir}/{self.__class__.__name__}/"
        os.makedirs(self.cache_dir, exist_ok=True)

        in_file = f"{self.cache_dir}/lambada_test.jsonl"
        # with utils.LockFile(self.cache_dir+"lock"):
        if not os.path.isfile(in_file):
            utils.download(self.URL, self.cache_dir, ignore_if_exists=True)

        with open(in_file, "r") as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]["text"]
        if self.sep is not None:
            d = self.sep + d
        d = self.vocabulary.sentence_to_indices(d)
        return {
            "data": np.array(d, dtype=self.dtype),
            "in_len": len(d),
            "index": idx
        }

    def start_test(self):
        return LambadaTest(self.data, vocabulary=self.vocabulary)
