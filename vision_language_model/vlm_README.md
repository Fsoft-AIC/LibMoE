# Vision Language Model Detail
## 🔍 Checkpoints

We are making our entire experiment checkpoints publicly available to contribute to the community's research on the topic of Mixture of Experts (MoE). By reusing our checkpoints at the **Pre-Training** and **Pre-FineTuning** stages, we hope to help others save time and computational resources in their own experiments.

| Stage            | Method       | Siglip 224 + Phi3.5 |
|:-----------------:|:-----------:|:--------------------:|
| **Pre-Training**  | —           | [Link](https://huggingface.co/Fsoft-AIC/Phi3.5-Siglip-MoE/tree/main/pretrain) |
| **Pre-FineTuning**| —           | [Link](https://huggingface.co/DavidNguyen/Phi3.5-Siglip-MoE/tree/main/pft) |
| **VIT 665K**      | SMoE      | [Link](https://huggingface.co/DavidNguyen/Phi3.5-Siglip-smoe-665K) |
|                   | XMoE        | [Link](https://huggingface.co/DavidNguyen/Phi3.5-Siglip-xmoe-665K/tree/main) |
|                   | SharedE-V2  | [Link](https://huggingface.co/DavidNguyen/Phi3.5-Siglip-smoe_share-665K/tree/main) |
|                   | SharedE-V3  | [Link](https://huggingface.co/DavidNguyen/Phi3.5-Siglip-smoe_sharev3-665K/tree/main) |
|                   | TC-MOE      | [Link](https://huggingface.co/DavidNguyen/Phi3.5-Siglip-smoe_tcmoe-665K/tree/main) |
|                   | MOE++       | [Link](https://huggingface.co/DavidNguyen/Phi3.5-Siglip-smoe_plus_plus-665K/tree/main) |
| **VIT 1M2**       | SMoE      | [Link](https://huggingface.co/DavidNguyen/Phi3.5-Siglip-smoe-1M2) |
|                   | XMoE        | [Link](https://huggingface.co/DavidNguyen/Phi3.5-Siglip-xmoe-1M2/tree/main) |
|                   | SharedE-V2  | [Link](https://huggingface.co/DavidNguyen/Phi3.5-Siglip-smoe_share-1M2/tree/main) |
|                   | SharedE-V3  | [Link](https://huggingface.co/DavidNguyen/Phi3.5-Siglip-smoe_sharev3-1M2/tree/main) |
|                   | TC-MOE      | [Link](https://huggingface.co/DavidNguyen/Phi3.5-Siglip-smoe_tcmoe-1M2/tree/main) |
|                   | MOE++       | [Link](https://huggingface.co/DavidNguyen/Phi3.5-Siglip-smoe_plus_plus-1M2/tree/main) |

*VIT stands for Visual Instruction Tuning.

## 🚀 Quick Start

### 📥 Installation

1. **Follow the unified repository setup** (see the root [Quick Start](../README.md#-quick-start)):

    ```bash
    pip install -e .
    pip install -e .[vlm,lm,eval]                # or: pip install -r requirements.txt
    # For a VLM-only environment after the base install:
    pip install -e .[vlm,eval]
    ```

2. **Install FlashAttention:**

    Choose the release that matches your CUDA / Torch build from the [FlashAttention releases](https://github.com/Dao-AILab/flash-attention/releases/). Example:

    ```bash
    pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu118torch2.1cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
    ```

### Troubleshooting: FusedAdam Issues

If you encounter errors related to `FusedAdam`, you can verify the CUDA module is loaded correctly:

```python
from deepspeed.ops.adam import FusedAdam
import importlib, pathlib

m = importlib.import_module("deepspeed.ops.adam.fused_adam_cuda")
print("FusedAdam CUDA loaded from:", pathlib.Path(m.__file__).resolve())
```

**If you encounter FusedAdam errors**, please refer to the troubleshooting guide: [Issue #1 - FusedAdam Fix](https://github.com/Fsoft-AIC/LibMoE/issues/1)

---
### 📊 Dataset Preparation
For a detailed, step-by-step guide on setting up the dataset, please refer to the [dataset guide](../docs/sparse_upcyling/dataset_guide.md).

### 🔧 Setup New MoE Layer
For a detailed step-by-step guide on setting up a new MoE layer, please refer to the [model guide](../docs/sparse_upcyling/model_guide.md).

### 🏋️‍♂️ Training

After downloading the datasets and the corresponding JSON files, you can proceed to train the model using the following commands. Below is an example using the `Phi3` configuration.

**Option 1: Run Each Stage Separately**

1. **Pre-train the MLP connector:**

    ```bash
    bash scripts/train/phi3mini/clip/pretrain_phi3.sh
    ```

2. **Pre-finetune the whole model:**

    ```bash
    bash scripts/train/phi3mini/clip/pft_phi3mini.sh
    ```

3. **Visual instruction tuning stage:**

    ```bash
    bash scripts/train/phi3mini/clip/sft_phi3mini.sh
    ```

**Option 2: Run All Stages**

You can run all stages in sequence with the following command:

```bash
bash scripts/train/run_train_all.sh
```

Note:
- These scripts are designed for training the model on a single node with 4x A100 GPUs.
- You must set the `batch_size` to the value specified in our scripts (`/scripts/train/phi3mini/clip`) for each stage (`batch_size = gradient_accumulation_steps * batch_size_current`).

**Test Training the Model**

We recommend running all stages with `MAX_STEPS=2` to check for issues in each stage. This approach allows you to identify and fix problems quickly, ensuring a stable process. After testing, set `MAX_STEPS=-1` to train all steps fully. Also, remember to delete the checkpoint folder that was created during testing.


```bash
#!/bin/bash
export TMPDIR=""
export TOOLKIT_DIR=""  # Path to the toolkitmoe directory
export KEY_HF=""       # Hugging Face API key
export ID_GPUS="0,1,2,3"
# Set to -1 to run all steps
export MAX_STEPS=2  # Select a suitable number of steps for testing each stage

echo "Starting pretrain stage"
bash ./scripts/train/phi3mini/pretrain_phi3.sh

echo "Starting pft stage"
bash ./scripts/train/phi3mini/pft_phi3mini.sh

echo "Starting sft stage"
bash ./scripts/train/phi3mini/sft_phi3mini.sh

```

### 🧪 Evaluation

We evaluate on multiple benchmarks:
- AI2D	
- ChartQA	
- Text VQA	
- GQA	
- HallusionBenchmark	
- MathVista Validation	
- MMBenchEN 
- MME	
- MMMU Validation	
- MMStar	
- POPE	
- SQA IMG Full

To run the evaluation, use the following command:

```bash
bash ./vision_language_model/scripts/eval/run_eval.sh
```

**Note**: For the MathVista Validation and HallusionBenchmark, GPT-4 is used for evaluation. You need to provide an API key to perform the evaluation.


---

#### ⚠️ Troubleshooting: Evaluation Hangs During Model Loading

If your evaluation process hangs or freezes when loading models/tokenizers (especially in multi-GPU setups), this is likely caused by the `filelock` mechanism in the Transformers library.

**Quick Fix**:
```bash
# Set these environment variables before running evaluation
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export TRANSFORMERS_OFFLINE=0

# Or clear stale lock files
find ~/.cache/huggingface -name "*.lock" -type f -delete
```

**For detailed solutions and troubleshooting steps**, see: [Issue #2 - Evaluation Hangs (FileLock)](https://github.com/Fsoft-AIC/LibMoE/issues/2)



### 📈 Expert Behavior Analysis

For comprehensive analysis of expert routing behavior, including post-hoc analysis of router behavior, expert selection patterns, and checkpoint trends, please refer to the [Expert Behavior Analysis Guide](../docs/sparse_upcyling/instruction_analysis_moe_vlm.md).

The guide covers:
- How to collect expert metrics during evaluation
- Configuring `return_id_experts` for logging
- Understanding output log formats
- Using the analysis toolkit in `vision_language_model/evaluate/analysis/`

**Note**: For other LLaVA variants, change the `conv_template` in `model_args`. Find the corresponding value in `conv_templates` dict in `moe_model/conversation.py`.
