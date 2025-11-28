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

This stage uses multiple vision-language datasets for comprehensive instruction tuning. Follow the steps below to download and prepare all required datasets.

### Expected Directory Structure

After completing all steps, your data directory should look like this:

```
libmoe/
└── data/
    ├── jsons/
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

---

### **Step 1: Install Requirements**

Before downloading, install the Hugging Face CLI:

```bash
pip install huggingface_hub
```

**Optional but highly recommended** for 5× faster downloads:

```bash
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
```

---

### **Step 2: Configure Download Paths**

Set up the download configuration:

```bash
# Hugging Face repository information
REPO_ID="DavidNguyen/LLAVA-LibMoE"
REPO_TYPE="dataset"

# Local path to store downloaded files
LOCAL_DIR="/mnt/d/workspace/libmoe"
mkdir -p "$LOCAL_DIR"
```

**If the dataset is private**, set your Hugging Face token:

```bash
export HF_TOKEN="hf_your_token_here"
```

> **Tip:** Get your token from [Hugging Face Settings](https://huggingface.co/settings/tokens)

---

### **Step 3: Download the Dataset**

Download all dataset files with automatic resume support:

**Standard download:**

```bash
huggingface-cli download "$REPO_ID" \
    --repo-type "$REPO_TYPE" \
    --local-dir "$LOCAL_DIR" \
    --token "$HF_TOKEN" \
    --resume-download
```

**Fast download** (if you installed `hf_transfer`):

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download "$REPO_ID" \
    --repo-type "$REPO_TYPE" \
    --local-dir "$LOCAL_DIR" \
    --token "$HF_TOKEN" \
    --resume-download
```

> **Note:** The download includes all `.zip` files and JSON annotation files. You can safely interrupt and resume the download at any time using `--resume-download`.

---

### **Step 4: Extract Dataset Files**

After downloading, extract all dataset archives into the correct LibMoE directory structure.

#### **4.1 Reconstruct Sharded Archives**

Some large datasets (OCR-VQA, OneVision) are split into multiple `.zip.part` files. First, concatenate them into complete `.zip` archives:

```bash
cd "$LOCAL_DIR"

# Reconstruct OCR-VQA archive
mkdir -p ./data/ocr_vqa
cat ./data/ocr_vqa/images_part_*.zip.part > ./data/ocr_vqa/images.zip

# Reconstruct OneVision archive
mkdir -p ./data/image_onevision
cat ./data/image_onevision/image_onevision_part_*.zip.part > ./data/image_onevision.zip
```

---

#### **4.2 Extract All Zip Files**

Now extract all `.zip` files while preserving the LibMoE directory structure:

```bash
echo "Extracting all .zip files..."

find "$LOCAL_DIR/data" -type f -name "*.zip" | while read -r file; do
    # Get the relative path of the zip file
    rel_path=$(dirname "${file#$LOCAL_DIR/data/}")
    
    # Determine target extraction directory
    if [ "$rel_path" = "." ]; then
        dst_dir="$LOCAL_DIR/data"
    else
        dst_dir="$LOCAL_DIR/data/$rel_path"
    fi
    
    echo "Extracting: $(basename "$file") → $dst_dir"
    mkdir -p "$dst_dir"
    unzip -q "$file" -d "$dst_dir"
done

echo "✓ All files extracted successfully!"
```

> **Note:** This script automatically handles all datasets and preserves the correct folder structure.

---

#### **4.3 Extraction Mapping**

The extraction process maps files to their correct locations:

| Source Archive | Extracted To |
|----------------|--------------|
| `data/coco/*.zip` | `libmoe/data/coco/train2017/` |
| `data/gqa/*.zip` | `libmoe/data/gqa/images/` |
| `data/ocr_vqa/images.zip` | `libmoe/data/ocr_vqa/images/` |
| `data/textvqa/*.zip` | `libmoe/data/textvqa/train_images/` |
| `data/vg/*.zip` | `libmoe/data/vg/VG_100K/` and `VG_100K_2/` |
| `data/image_onevision.zip` | `libmoe/data/image_onevision/` |

---

### **Step 5: Verify Installation**

Verify that all datasets are properly extracted:

```bash
cd "$LOCAL_DIR"
tree data -L 2 -d
```

**Expected directory structure:**

```
data/
├── jsons/
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

**Check image counts** (optional):

```bash
echo "COCO: $(find data/coco/train2017 -type f | wc -l) images"
echo "GQA: $(find data/gqa/images -type f | wc -l) images"
echo "OCR-VQA: $(find data/ocr_vqa/images -type f | wc -l) images"
echo "TextVQA: $(find data/textvqa/train_images -type f | wc -l) images"
echo "VG: $(find data/vg -type f | wc -l) images"
```

---

### Dataset Sources Reference

All datasets are sourced from their original repositories:

| Dataset | Original Source | Notes |
|---------|----------------|-------|
| **COCO** | [train2017](http://images.cocodataset.org/zips/train2017.zip) | 118K images |
| **GQA** | [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip) | Scene graph QA |
| **OCR-VQA** | [Google Drive](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing) | Converted to `.jpg` format |
| **TextVQA** | [train_val_images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip) | Text reading in images |
| **Visual Genome** | [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip) | Dense annotations |
| **OneVision** | [HuggingFace](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data) | Multi-modal data |

---

## **Stage 4: Training Script Example**

Once all datasets are prepared, you can start training. Use environment variables for flexible path configuration.

### Example Training Command

```bash
# Set the data directory
export TOOLKIT_DIR="$LOCAL_DIR"

# Launch training with DeepSpeed
deepspeed --include localhost:$ID_GPUS moe_model/train/train_mem.py \
    --deepspeed ./scripts/zero3_offload.json \
    --model_name_or_path microsoft/Phi-3.5-mini-instruct \
    --version phi35 \
    --data_path "$TOOLKIT_DIR/data/jsons/llava_v1_5_mix665k_half.json" \
    --image_folder "$TOOLKIT_DIR/data"
```

> **Tip:** Adjust `$ID_GPUS` to specify which GPUs to use (e.g., `0,1,2,3` for 4 GPUs).

---

## **Troubleshooting**

<details>
<summary><b>Download interrupted or failed</b></summary>

Simply re-run the download command with `--resume-download`. The Hugging Face CLI will automatically resume from where it stopped.

```bash
huggingface-cli download "$REPO_ID" \
    --repo-type "$REPO_TYPE" \
    --local-dir "$LOCAL_DIR" \
    --resume-download
```

</details>

<details>
<summary><b>Extraction fails with "unzip: command not found"</b></summary>

Install unzip:

```bash
# Ubuntu/Debian
sudo apt-get install unzip

# macOS
brew install unzip
```

</details>

<details>
<summary><b>Permission denied errors</b></summary>

Ensure you have write permissions to the target directory:

```bash
chmod -R u+w "$LOCAL_DIR"
```

</details>

---
