# 📊 Dataset Preparation Guide for LibMoE

This guide explains **how to prepare all datasets and images** for training a vision-language Mixture-of-Experts (MoE) model using LibMoE and LLaVA-style data.

---

## **Stage 1: Pre-Training**

**Pre-train the MLP connector** using the [LLaVA-558K](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) dataset.

<details>
<summary>Download Commands</summary>

```bash
mkdir -p ./data/jsons

# Download JSON annotation files
wget -P ./data/jsons https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/blip_laion_cc_sbu_558k.json?download=true

# Download and extract image files
wget -P ./data https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip?download=true
unzip ./data/images.zip -d ./data/images
rm ./data/images.zip
```

</details>

---

## **Stage 2: Pre-FineTuning**

**Warm up your model** with the [ALLaVA-4V](https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V) caption data.

* Download directly from Hugging Face:
  [FreedomIntelligence/ALLaVA-4V](https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V)

---

## **Stage 3: Visual Instruction Tuning**


* Manually download the required datasets.

#### **Download Dataset JSON Files**

```bash
mkdir -p ./data/jsons

# Visual instruction tuning datasets
wget -P ./data/jsons https://huggingface.co/datasets/DavidNguyen/LLAVA-LibMoE/resolve/main/llava_v1_5_mix665k.json?download=true
wget -P ./data/jsons https://huggingface.co/datasets/DavidNguyen/LLAVA-LibMoE/resolve/main/llava_v1_5_mix665k_half.json?download=true
wget -P ./data/jsons https://huggingface.co/datasets/DavidNguyen/LLAVA-LibMoE/resolve/main/data_1M2_correct_format.json?download=true
```

**Datasets Used:**

* [OneVision-1M2](https://huggingface.co/datasets/DavidNguyen/LLAVA-LibMoE)
* [LLaVA-665K](https://huggingface.co/datasets/DavidNguyen/LLAVA-LibMoE)
* [LLaVA-332K](https://huggingface.co/datasets/DavidNguyen/LLAVA-LibMoE)

---

## **Stage 4: Image Preparation**

**Download and organize the image files for all components:**

| Dataset          | Download Link / Instructions                                                                                                                                                                                                                                       |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **COCO**         | [train2017.zip](http://images.cocodataset.org/zips/train2017.zip)                                                                                                                                                                                                  |
| **GQA**          | [images.zip](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)                                                                                                                                                                                            |
| **OCR-VQA**      | [Google Drive folder (save as .jpg)](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing)                                                                                                                                         |
| **TextVQA**      | [train_val_images.zip](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)                                                                                                                                                                         |
| **VisualGenome** | [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)                                                                                                                          |
| **OneVision**    | [image_onevision directory](https://huggingface.co/datasets/DavidNguyen/LLAVA-LibMoE/tree/main/data/image_onevision) <br> Use the provided [process.sh](https://huggingface.co/datasets/DavidNguyen/LLAVA-LibMoE/blob/main/data/image_onevision/process.sh) script |

**For OneVision:**

1. Download all piece files and `process.sh`.
2. Run `process.sh` to combine pieces and unzip.
3. Images will be inside `image_onevision/`.

---

### **Directory Structure Example**

Organize your data as follows (relative to the project root):

```plaintext
libmoe/
└── data/
    ├── image_onevision/
    ├── coco/
    │   └── train2017/
    ├── gqa/
    │   └── images/
    ├── ocr_vqa/
    │   └── images/
    ├── textvqa/
    │   └── train_images/
    └── vg/
        ├── VG_100K/
        └── VG_100K_2/
```

* **LLaVA-665K/332K** images are drawn from: `coco/`, `gqa/`, `ocr_vqa/`, `textvqa/`, `vg/`
* **OneVision-1M2** images are in: `image_onevision/`

---

## **Stage 5: Training Script Example**

When running your training scripts, specify paths via environment variables for flexibility.
For example, to train with **LLaVA-332K** and your data in `./libmoe/data/`:

```bash
export TOOLKIT_DIR=./libmoe

deepspeed --include localhost:$ID_GPUS moe_model/train/train_mem.py \
    --deepspeed ./scripts/zero3_offload.json \
    --model_name_or_path microsoft/Phi-3.5-mini-instruct \
    --version phi35 \
    --data_path $TOOLKIT_DIR/data/jsons/llava_v1_5_mix665k_half.json \
    --image_folder $TOOLKIT_DIR/data
```

---
