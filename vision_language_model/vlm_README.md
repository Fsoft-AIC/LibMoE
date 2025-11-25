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

2. **Install FlashAttention (optional but recommended):**

    Choose the release that matches your CUDA / Torch build from the [FlashAttention releases](https://github.com/Dao-AILab/flash-attention/releases/). Example:

    ```bash
    pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu118torch2.1cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
    ```
### 📊 Dataset Preparation
For a detailed, step-by-step guide on setting up the dataset, please refer to the [dataset guide](https://github.com/Fsoft-AIC/LibMoE/blob/main/docs/sparse_upcyling/dataset_guide.md).

### 🔧 Setup New MoE Layer
For a detailed step-by-step guide on setting up a new MoE layer, please refer to the [model guide](https://github.com/Fsoft-AIC/LibMoE/blob/main/docs/sparse_upcyling/model_guide.md).

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
We are evaluate multi-benchmark
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
bash scripts/eval/run_eval.sh
```
*Note: For the MathVista Validation and HallusionBenchmark, GPT-4 is used for evaluation. You need to provide an API key to perform the evaluation.

### Multiple Usages

**Evaluation of LLaVA on MME**

```bash
python3 -m accelerate.commands.launch \
    --num_processes=8 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.5-7b" \
    --tasks mme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.5_mme \
    --output_path ./logs/ \
    --return_id_experts true \  # return selected expert IDs
    --layers_expert_selection 1 2 3  # define specific layers for expert selection; if no layer IDs are defined, all experts from all layers are selected by default
```

**Evaluation of LLaVA on multiple datasets**

```bash
python3 -m accelerate.commands.launch \
    --num_processes=8 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.5-7b" \
    --tasks mme,mmbench_en \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.5_mme_mmbenchen \
    --output_path ./logs/ \
    --return_id_experts true \  # return selected expert IDs
    --layers_expert_selection 1 2 3  # define specific layers for expert selection; if no layer IDs are defined, all experts from all layers are selected by default
```

### 📈 Analyst Tools
For post-hoc analysis of router behaviour, expert selection, and checkpoint trends, use the refactored toolkit documented in `vision_language_model/evaluate/analysis/analyst_README.md`. It covers the CLI modules, notebook runners, and data artefacts that underpin the evaluation dashboards.

**For other variants llava. Please change the `conv_template` in the `model_args`**

> `conv_template` is an arg of the init function of llava in `lmms_eval/models/llava.py`, you could find the corresponding value at LLaVA's code, probably in a dict variable `conv_templates` in `moe_model/conversation.py`

```bash
python3 -m accelerate.commands.launch \
    --num_processes=8 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-mistral-7b,conv_template=mistral_instruct" \
    --tasks mme,mmbench_en \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.5_mme_mmbenchen \
    --output_path ./logs/ \
    --return_id_experts true \  # return selected expert IDs
    --layers_expert_selection 1 2 3  # define specific layers for expert selection; if no layer IDs are defined, all experts from all layers are selected by default
```
