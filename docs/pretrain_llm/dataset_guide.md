# Dataset Guide (Pretrain LLM)

This project streams SlimPajama shards from Hugging Face and incrementally tokenizes/caches only what’s needed for your run. You can change the dataset or plug in your own with a small subclass.

## Quick Start

- Ensure dependencies include `requests`, `zstandard` and `sentencepiece` (install if missing).
- Optionally copy a prebuilt SentencePiece model into `cache` to reproduce exact tokenization; otherwise the tokenizer will be trained on-the-fly from streamed data.
- Launch a SlimPajama run (examples use default 8k vocab and 100-token unroll):
  - Small (class `SlimPajama`): `-task slimpajama_transformer`
  - Large (class `SlimPajamaLarge`): `-task slimpajama_large_transformer`

```bash
torchrun --nproc_per_node=8 pretrain_language_modeling/main.py \
  -task slimpajama_transformer \
  -batch_size 8 -lm.unroll 1024 -sentencepiece.n_pieces 8000 \
  -stop_after 10000   # optional: limits total training steps and tokenization
```

## SlimPajama Datasets

- Training and validation are provided by two dataset classes:
  - `pretrain_language_modeling/framework/dataset/text/slimpajama.py` (class `SlimPajama`)
  - `pretrain_language_modeling/framework/dataset/text/slimpajama_large.py` (class `SlimPajamaLarge`)
- Both inherit `ChunkedSentencepieceLMDataset` and:
  - Stream JSONL `.zst` shards from Hugging Face.
  - Train or load a SentencePiece tokenizer and tokenize into binary chunk files under `cache/`.
  - Only tokenize enough data for your configured run (respects `-stop_after`, `-batch_size`, and `-lm.unroll`).

Notes:
- Validation size is controlled by `-lmds.valid_ratio` (default 1.0) in the task files.
- A special token `<STORY_SEP>` is appended between documents during tokenization to reduce cross-doc leakage.

## Tokenizer And Cache Layout

- The dataset stores artifacts under `cache/{DatasetClass}/{Variant}/`, where `Variant` encodes vocab size (e.g., `SlimPajama-8000`).
- Tokenizer path is generated at runtime. If you want to pin a specific tokenizer, place its model file in the dataset cache directory before first run so retraining is skipped.
- If you replicate an existing cache, copy the entire `cache/` tree to preserve tokenizers and tokenized chunks exactly.

## Key Configuration Flags

- `-sentencepiece.n_pieces` (default 8000): vocabulary size for SentencePiece tokenizer.
- `-lm.unroll`: tokens per training sample (sequence length). Use the same value for eval via `-lm.unroll_eval` or leave it to default to `-lm.unroll`.
- `-stop_after`: total training iterations; also used by dataset tasks to compute a tokenization limit so you don’t over-tokenize.
- `-lmds.valid_ratio` (task-specific): scales the number of validation tokens processed.
- `-fs_cache_pattern`: optional filesystem cache for binary chunks on local disks (see below).

## Filesystem Cache (Optional)

- If training reads tokenized chunks from a network filesystem, you can stage them to a local disk cache for faster I/O.
- Pass `-fs_cache_pattern "*"` (default) to enable probing local writable mounts; set to `none` to disable.
- The cache mirrors files under a per-user path on the fastest local disk and is transparent to the DataLoader.

## Add A Custom Dataset

The easiest path is to subclass `ChunkedSentencepieceLMDataset` and implement two methods:

```python
class MyTextDataset(ChunkedSentencepieceLMDataset):
    def get_url(self, index: int, split: Optional[str] = None) -> str:
        # Return an http(s) URL, or a local file via 'file:///abs/path/shard_{index}.jsonl.zst'
        ...

    def get_n_shards(self, split: Optional[str] = None) -> int:
        # Return the number of shards available for this split
        ...
```

Optional customizations:
- Override `parse(self, line: str) -> str` if your JSONL schema differs (default expects a `{"text": ...}` field).
- Return fewer samples for tokenizer training by overriding `TOKENIZER_N_FILES` or `get_tokenizer_n_files()`.
- If you need gzip instead of zstd, emit `.gz` in URLs; the loader handles `.zst` and `.gz` automatically.

Once implemented, expose it in `framework/dataset/__init__.py`, create a task file under `pretrain_language_modeling/tasks/`, and run with `-task your_task_name`.

## Local/Offline Data

- You can point `get_url` to local files by returning a `file:///...` URL. The dataset supports both `.zst` and `.gz` compressed JSONL streams.
- For fully offline runs, ensure the tokenizer model exists in the dataset cache directory so the code does not attempt to train it from streamed data.

## Modify Built-in Datasets

- Training datasets:
  - Update `pretrain_language_modeling/framework/dataset/text/slimpajama.py` or `slimpajama_large.py` to tweak shard lists or sampling order.
  - The constants at the top (`CHUNK_SIZES`, `TYPE_MAP`, `_DATA_URL`) control which shards are used and how URLs are formed.
- Evaluation datasets:
  - Edit or add files under `pretrain_language_modeling/framework/dataset/text/eval/` (e.g., `hellaswag.py`).

## Troubleshooting

- Missing zstandard or sentencepiece: install with `pip install zstandard sentencepiece`.
- Slow first epoch: tokenizer training and initial tokenization are performed once per cache; subsequent runs reuse artifacts.
- Tokenizer reproducibility: to avoid minor differences across SentencePiece versions, copy the project’s `cache/` directory (including tokenizer and tokenized chunks) between machines.
- Token limit not honored: check `-stop_after`, `-batch_size`, and `-lm.unroll`—tasks compute a token budget from these values and pre-tokenize just enough chunks.
