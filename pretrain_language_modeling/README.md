# Pretrain Language Modeling (LibMoE v2)

This package contains the training code to pretrain language models with Mixture‑of‑Experts (MoE) variants. It includes a configurable Transformer, multiple MoE layers, Triton kernels for fast expert routing, and streaming datasets with on-the-fly tokenization.

## Highlights
- **Multiple MoE implementations** under `pretrain_language_modeling/layers` (select via `MOE_TYPE`)
- **Triton‑based sparse batched matmul** in `layers/cvmm.py` for efficient expert routing
- **Streaming SlimPajama datasets** with on‑the‑fly SentencePiece tokenization and caching
- **Flexible task system** that wires models and datasets for quick experimentation
- **Built-in MoE variants**: vanilla MoE, X-MoE, DeepSeek-v2/v3, and more research variants

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Layout](#project-layout)
- [MoE Variants](#moe-variants)
- [Configuration](#configuration)
- [Datasets](#datasets)
- [Training and Evaluation](#training-and-evaluation)
- [Further Reading](#further-reading)

## Installation

Install required dependencies:
```bash
pip install -r requirements.txt
```

**GPU Requirements:** Triton kernels require a compatible CUDA GPU and driver. Kernels JIT-compile on first use.

## Quick Start

The easiest way to get started is to use the provided scripts in `pretrain_language_modeling/scripts/`:

```bash
# Training
bash pretrain_language_modeling/scripts/train.sh

# Evaluation
bash pretrain_language_modeling/scripts/eval.sh
```

See [Training and Evaluation](#training-and-evaluation) section for how to customize these scripts.

## Project Layout

```
pretrain_language_modeling/
├── layers/              # MoE layers and Triton kernels
│   ├── cvmm.py         # Sparse batched matmul (Triton)
│   ├── moe_layer.py    # Vanilla MoE implementation
│   ├── moe_layer_xmoe.py          # X-MoE variant
│   ├── moe_layer_deepseekv2.py    # DeepSeek-v2 variant
│   └── ...             # Other MoE variants
├── models/             # Transformer language model
├── tasks/              # Task definitions (datasets + model wiring)
├── framework/
│   └── dataset/        # Dataset building blocks
│       └── text/       # SlimPajama and other text datasets
├── sweeps/             # Example experiment configs (YAML)
├── scripts/            # Example run scripts
└── main.py             # Main training entrypoint
```

## MoE Variants

The framework supports multiple MoE implementations that can be selected via the `MOE_TYPE` environment variable:

| Variant | Description |
|---------|-------------|
| `moe_layer` (default) | Vanilla MoE with linear router and ReLU experts |
| `moe_layer_xmoe` | Cosine router variant (X-MoE style) with temperature |
| `moe_layer_deepseekv2` | DeepSeek-v2 style with shared keys/values |
| `moe_layer_deepseekv3` | DeepSeek-v3 variant with additional regularization |
| `moe_layer_remoe` | ReMoE research variant |
| `moe_layer_plus_plus` | MoE++ research variant |
| `moe_layer_tc_moe` | TC-MoE research variant |

**How to select:** Set `MOE_TYPE` environment variable in your training script (e.g., `export MOE_TYPE="moe_layer_xmoe"`).

**For detailed MoE architecture, parameters, and how to add custom variants**, see [Model Guide](../docs/pretrain_llm/model_guide.md).

## Configuration

Model and training configurations are managed through:
- **YAML config files** in `pretrain_language_modeling/sweeps/` (e.g., `154M/`, `660M/`)
- **Command-line arguments** that can override YAML settings

Common configurations include model size, batch size, learning rate, sequence length, MoE parameters (number of experts, expert size, routing strategy), and training duration.

**For complete list of arguments and hyperparameters**, see [Model Guide](../docs/pretrain_llm/model_guide.md).

## Datasets

The framework uses **SlimPajama** datasets that stream shards from Hugging Face and perform on-the-fly SentencePiece tokenization. Tokenized data is cached under `cache/` directory for reuse.

**Key features:**
- Streaming JSONL `.zst` shards with incremental tokenization
- Automatic tokenizer training on first run
- Smart caching that only tokenizes what's needed for your run
- Support for custom datasets via subclassing

**For dataset setup, tokenization details, and adding custom datasets**, see [Dataset Guide](../docs/pretrain_llm/dataset_guide.md).

## Training and Evaluation

### Training

Use the provided training script in `pretrain_language_modeling/scripts/train.sh`:

<augment_code_snippet path="pretrain_language_modeling/scripts/train.sh" mode="EXCERPT">
````bash
export MOE_TYPE="moe_layer"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
...
torchrun --master_port $MASTER_PORT --nproc_per_node=$NUM_DEVICES run.py  \
    /path/to/config.yaml
````

**To customize your training:**

1. **Select MoE variant**: Change `MOE_TYPE` (e.g., `moe_layer_xmoe`, `moe_layer_deepseekv2`)
2. **Set GPUs**: Modify `CUDA_VISIBLE_DEVICES` to specify which GPUs to use
3. **Choose config**: Point to a YAML config in `pretrain_language_modeling/sweeps/`
   - `154M/` — smaller models for quick experiments
   - `660M/` — larger models for production runs
4. **Adjust config**: Edit the YAML file or override via command-line arguments

**Example modifications:**
```bash
# Use DeepSeek-v2 variant with 2 GPUs
export MOE_TYPE="moe_layer_deepseekv2"
export CUDA_VISIBLE_DEVICES="0,1"

# Point to your chosen config
torchrun --nproc_per_node=2 run.py \
    pretrain_language_modeling/sweeps/154M/slimpajama_moe_no_attmoe_154M_deepseekv2.yaml
```

### Evaluation

Use the provided evaluation script in `pretrain_language_modeling/scripts/eval.sh`:

````bash
export MOE_TYPE="moe_layer_deepseek"
...
tasks=("lambada" "cbt" "hellaswag" "piqa" ...)
checkpoint_path="/path/to/checkpoint"
````
**To customize your evaluation:**

1. **Select MoE variant**: Change `MOE_TYPE` to match your trained model
2. **Set checkpoint path**: Point `checkpoint_path` to your saved checkpoint (file or directory)
3. **Choose tasks**: Modify the `tasks` array to select evaluation benchmarks
4. **Adjust batch size**: Change `bs` variable for evaluation batch size

**Available evaluation tasks:**
- `lambada`, `cbt`, `hellaswag`, `piqa`, `blimp`, `ai2arc`
- `mmlu`, `openbookqa`, `winogrande`, `siqa`, `commonsenseqa`, `race`

## Further Reading

For detailed documentation on specific topics:

- **[Model Guide](../docs/pretrain_llm/model_guide.md)** — MoE architecture, parameters, adding custom variants, Triton utilities, and complete hyperparameter reference
- **[Dataset Guide](../docs/pretrain_llm/dataset_guide.md)** — Dataset setup, tokenization, caching, adding custom datasets, and configuration options
- **[Checkpoint List](../docs/pretrain_llm/checkpoint_list.md)** — Pre-trained model checkpoints and configurations
