# Tool Kit MoE

## Contents
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)

## Installation

1. **Clone this repository:**

    ```bash
    git clone https://gitlab.com/DAVID-NGUYEN-S16/toolkitmoe.git
    cd toolkitmoe
    ```

2. **Install dependencies:**

    *We used Python 3.9 `venv` for all experiments, and it should be compatible with Python 3.9 or 3.10 under Anaconda if you prefer to use it.*

    - Using `venv`:

        ```bash
        python -m venv /path/to/new/virtual/moe
        source /path/to/new/virtual/moe/bin/activate
        ```

    - Using `Anaconda`:

        ```bash
        conda create -n test python=3.9 -y
        conda activate test
        ```

    Then, install the required packages:

    ```bash
    pip install --upgrade pip
    pip install -e .
    pip install -r ./requirements.txt
    ```

3. **Install additional packages:**

    Choose the FlashAttention version based on your Torch version from the [FlashAttention Releases](https://github.com/Dao-AILab/flash-attention/releases/).

    Example:

    ```bash
    pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu118torch2.1cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
    ```

## Dataset Preparation

**Stage 1: Pre-Training**

For pre-training, we use the [LLaVA-558K](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) dataset to pretrain the MLP connector.

**Stage 2: Pre-FineTuning**

For pre-finetuning, we use the [ALLaVA](https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V) caption data to warm-up the model.

**Stage 3: Visual Instruction Tuning**

For the visual instruction tuning stage, we use a mixture of datasets:

- [LLaVA-665K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json)
- [LLaVA-332K](https://huggingface.co/datasets/DavidNguyen/LLAVAHALF/blob/main/llava_v1_5_mix665k_half.json)


We are using LLaVA-332K for our experiments. You can set the `$TOOLKIT_DIR` environment variable to specify the path to the parent directory of the project root (e.g., `cm/namnv/toolkitmoe` -> `TOOLKIT_DIR=cm/namnv`).

## Setup New MoE Layer

For a detailed step-by-step guide on setting up a new MoE layer, please refer to the [model guide](https://gitlab.com/DAVID-NGUYEN-S16/toolkitmoe/-/blob/main/docs/model_guide.md).

## Training

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

## Evaluation
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
    --output_path ./logs/
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
    --output_path ./logs/
```

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
    --output_path ./logs/
```












