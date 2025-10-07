## Contents
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)

## Dataset Preparation
### Stage 1: Pre-Training
For pre-training, we use the [LLaVA-558K](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) to pretrain the mlp connector.

### Stage 2: Pre-FineTuning
For pre-finetuning, we use the [ALLaVA](https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V) caption data to warm-up the whole CuMo model.

### Stage 3: Visual Instruction Tuning
For the visual intruction tuning stage, we use the a mixture of datasets for training:

- [LLaVA-665K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json)


You can set $TOOLKIT_DIR to specify the path to the root directory of the project.

## Training
After downloading the datasets and the JSON files, you can proceed to train the model using the following commands. Taking Phi3 as an example, the first step is to pre-train the MLP connector.
```bash
bash scripts/cumo/mistral_7b/pretrain_mistral_7b.sh
```

The next step is to pre-finetune the whole model,
```bash
bash scripts/cumo/mistral_7b/pft_mistral_7b.sh
```

The final step is the visual instruction tuning stage,
```bash
bash scripts/cumo/mistral_7b/sft_mistral_7b.sh
```

Note that these scripts are for training the model on a single node of 8xA100s. If you want to train the model on multiple nodes, you can use the [deepspeed](https://www.deepspeed.ai/getting-started/) multi-node trainings with added hostfile in the scripts.

## Evaluation
We evaluate CuMo models on multiple benchmarks and many scripts are based on [LLaVA](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md) evaluation settings. We've adapted some of them into multi-GPU evaluation scripts and added evaluations on MMMU and Mathvista. You can download the checkpoints for CuMo [mistral-7b](https://huggingface.co/shi-labs/CuMo-mistral-7b) / [mixtral-8x7b](https://huggingface.co/shi-labs/CuMo-mixtral-8x7b) models and follow the evaluation instructions in [LLaVA](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md) to download the datasets accordingly. The datasets are structured as:

```

In










