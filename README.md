# LibMoE: A LIBRARY FOR COMPREHENSIVE BENCHMARKING MIXTURE OF EXPERTS IN LARGE LANGUAGE MODELS
**Authors:** Nam V. Nguyen, Thong T. Doan, Luong Tran, Van Nguyen, Quang Pham

<p align="center">
  <a href="https://arxiv.org/abs/2411.00918">
    <img src="https://img.shields.io/badge/arXiv-2411.00918-red?style=flat&label=arXiv">
  </a>
  <a href="https://fsoft-aic.github.io/fsoft-LibMoE.github.io/">
    <img src="https://custom-icon-badges.demolab.com/badge/WebPage-1a4f76?style=flat&logo=web">
  </a>
</p>


<p align="center">
    <a href="https://github.com/Fsoft-AIC/LibMoE?tab=readme-ov-file#-release-notes" style="text-decoration: none;">📢 Release Notes</a> •
    <a href="https://github.com/Fsoft-AIC/LibMoE?tab=readme-ov-file#-quick-start" style="text-decoration: none;">🚀 Quick Start</a> •
    <a href="https://github.com/Fsoft-AIC/LibMoE?tab=readme-ov-file#-about" style="text-decoration: none;">📌 About</a> •
    <a href="https://github.com/Fsoft-AIC/LibMoE/blob/main/docs/model_guide.md" style="text-decoration: none;">🔧 Setup New MoE Layer</a> •
    <a href="https://github.com/Fsoft-AIC/LibMoE?tab=readme-ov-file#%EF%B8%8F%EF%B8%8F-training" style="text-decoration: none;">🏋️‍♂️ Training</a> •
    <a href="https://github.com/Fsoft-AIC/LibMoE?tab=readme-ov-file#-evaluation" style="text-decoration: none;">🧪 Evaluation</a> •
    <a href="https://github.com/Fsoft-AIC/LibMoE?tab=readme-ov-file#-citation" style="text-decoration: none;">📌 Citation</a>
    
</p>

## 📌 About
Mixture of Experts (MoEs) plays an important role in the development of more efficient and effective large language models (LLMs). Due to the enormous resource requirements, studying large scale MoE algorithms remain in-accessible to many researchers. This work develops LibMoE, a comprehensive and modular framework to streamline the research, training, and evaluation of MoE algorithms. Built upon three core principles: (i) modular design, (ii) efficient training; (iii) comprehensive evaluation, LibMoE brings MoE in LLMs more accessible to a wide range of researchers by standardizing the training and evaluation pipelines. 
Using LibMoE, we extensively benchmarked five state-of-the-art MoE algorithms over three different LLMs and 11 datasets under the zero-shot setting. The results show that despite the unique characteristics, all MoE algorithms perform roughly similar when averaged across a wide range of tasks. With the modular design and extensive evaluation, we believe LibMoE will be invaluable for researchers to make meaningful progress towards the next generation of MoE and LLMs.
## 📢 Release Notes

| Date           | Release Notes                                                                                              |
|----------------|------------------------------------------------------------------------------------------------------------|
| 2024-11-04 | - Additional feature metric analysis for MoE algorithms in the [LibMoE]() paper – ✅ |
| 2024-11-01 | - Released LibMoE v1.0 preprint report [HERE](https://arxiv.org/pdf/2411.00918) ✅  <br> - LibMoE webpage [HERE](https://fsoft-aic.github.io/fsoft-LibMoE.github.io/)  ✅  <br> - Publicly available checkpoints ✅ |


## 🔍 Checkpoints

We are making our entire experiment checkpoints publicly available to contribute to the community's research on the topic of Mixture of Experts (MoE). By reusing our checkpoints at the **Pre-Training** and **Pre-FineTuning** stages, we hope to help others save time and computational resources in their own experiments.

|      Method     | Stage              | Siglip 224 + Phi3.5 | Siglip 224 + Phi3 | CLIP 336 + Phi3|
|:---------------:|:------------------:|:---------------------------------------------------------:|:-------------------------------------------------------:|:-----------------------------------------------------:|
| **Pre-Training**  |                    | [Link](https://huggingface.co/Fsoft-AIC/Phi3.5-Siglip-MoE/tree/main/pretrain)                | [Link](https://huggingface.co/Fsoft-AIC/Phi3-SigLiP-MoE/tree/main/pretrain)              | [Link](https://huggingface.co/Fsoft-AIC/Phi3-CLIP-MoE/tree/main/pft)            |
| **Pre-FineTuning** |                    | [Link](https://huggingface.co/Fsoft-AIC/Phi3.5-Siglip-MoE/tree/main/pft)                | [Link](https://huggingface.co/Fsoft-AIC/Phi3-SigLiP-MoE/tree/main/pft)              | [Link](https://huggingface.co/Fsoft-AIC/Phi3-CLIP-MoE/tree/main/pretrain)            |
| **VIT 665K**    | SMoE-R             | [Link](https://huggingface.co/Fsoft-AIC/Phi3.5-Siglip-MoE/tree/main/sft_full/smoe)                | [Link](https://huggingface.co/Fsoft-AIC/Phi3-SigLiP-MoE/tree/main/sft/smoe)              | [Link](https://huggingface.co/Fsoft-AIC/Phi3-CLIP-MoE/tree/main/sft_full/smoe)            |
|                 | Cosine-R           | [Link](https://huggingface.co/Fsoft-AIC/Phi3.5-Siglip-MoE/tree/main/sft_full/smoe_cosinegating)                | [Link](https://huggingface.co/Fsoft-AIC/Phi3-SigLiP-MoE/tree/main/sft/smoe_cosinegating)              | [Link](https://huggingface.co/Fsoft-AIC/Phi3-CLIP-MoE/tree/main/sft_full/smoe_cosinegating)            |
|                 | Sigmoid-R          | [Link](https://huggingface.co/Fsoft-AIC/Phi3-SigLiP-MoE/tree/main/sft)                | [Link](https://huggingface.co/Fsoft-AIC/Phi3-SigLiP-MoE/tree/main/sft)              | [Link](https://huggingface.co/Fsoft-AIC/Phi3-SigLiP-MoE/tree/main/sft)            |
|                 | Hyper-R            | [Link](https://huggingface.co/Fsoft-AIC/Phi3.5-Siglip-MoE/tree/main/sft_full/hyperrouter)                | [Link](https://huggingface.co/Fsoft-AIC/Phi3-SigLiP-MoE/tree/main/sft/hyperrouter)              | [Link](https://huggingface.co/Fsoft-AIC/Phi3-CLIP-MoE/tree/main/sft_full/hyperrouter)            |
|                 | Perturbed Cosine-R | [Link](https://huggingface.co/Fsoft-AIC/Phi3.5-Siglip-MoE/tree/main/sft_full/smoe_perturbed)                | [Link](https://huggingface.co/Fsoft-AIC/Phi3-SigLiP-MoE/tree/main/sft/smoe_perturbed)              | [Link](https://huggingface.co/Fsoft-AIC/Phi3-CLIP-MoE/tree/main/sft_full/smoe_perturbed)            
---
*VIT stands for Visual Instruction Tuning.

[more ...](https://github.com/Fsoft-AIC/LibMoE/blob/main/docs/checkpoint_list.md)
## 🚀 Quick Start

### 📥 Installation

1. **Clone this repository:**

    ```bash
    git clone https://github.com/Fsoft-AIC/LibMoE.git
    cd LibMoE
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
        conda create -n moe python=3.9 -y
        conda activate moe
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
### 📊 Dataset Preparation
For a detailed, step-by-step guide on setting up the dataset, please refer to the [dataset guide](https://github.com/Fsoft-AIC/LibMoE/blob/main/docs/dataset_guide.md).

### 🔧 Setup New MoE Layer
For a detailed step-by-step guide on setting up a new MoE layer, please refer to the [model guide](https://github.com/Fsoft-AIC/LibMoE/blob/main/docs/model_guide.md).

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

## 📌 Citation
If you find this repository useful, please consider citing our paper:

```
@misc{nguyen2024libmoelibrarycomprehensivebenchmarking,
      title={LIBMoE: A Library for comprehensive benchmarking Mixture of Experts in Large Language Models}, 
      author={Nam V. Nguyen and Thong T. Doan and Luong Tran and Van Nguyen and Quang Pham},
      year={2024},
      eprint={2411.00918},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.00918}, 
}
```