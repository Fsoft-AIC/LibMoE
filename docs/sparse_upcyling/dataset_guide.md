# 📊 Dataset Preparation

## Stage 1: Pre-Training

For pre-training, we use the [LLaVA-558K](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) dataset to pretrain the MLP connector.

```bash
mkdir -p ./data/jsons
wget -P ./data/jsons https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/blip_laion_cc_sbu_558k.json?download=true
wget -P ./data/jsons https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/blip_laion_cc_sbu_558k.json?download=true
wget -P ./data https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip?download=true
unzip ./data/images.zip -d ./data/images
rm ./data/images.zip
```

## Stage 2: Pre-FineTuning

For pre-finetuning, we use the [ALLaVA](https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V) caption data to warm up the model.

## Stage 3: Visual Instruction Tuning



For the visual instruction tuning stage, we use a combination of datasets:
- [OneVision-1M2](https://huggingface.co/datasets/DavidNguyen/LLAVA-LibMoE)
- [LLaVA-665K](https://huggingface.co/datasets/DavidNguyen/LLAVA-LibMoE)
- [LLaVA-332K](https://huggingface.co/datasets/DavidNguyen/LLAVA-LibMoE)



```bash
wget -P ./data/jsons https://huggingface.co/datasets/DavidNguyen/LLAVA-LibMoE/resolve/main/llava_v1_5_mix665k.json?download=true
wget -P ./data/jsons https://huggingface.co/datasets/DavidNguyen/LLAVA-LibMoE/resolve/main/llava_v1_5_mix665k_half.json?download=true
```

### Image Preparation

Please download the image files from the respective datasets:

- COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip)
- GQA: [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
- OCR-VQA: [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing) - save files as `.jpg`
- TextVQA: [train_val_images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
- VisualGenome: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)

After downloading, organize the data as follows in the `./libmoe/data` directory:

```plaintext
libmoe
└── data
    ├── coco
    │   └── train2017
    ├── gqa
    │   └── images
    ├── ocr_vqa
    │   └── images
    ├── textvqa
    │   └── train_val_images
    └── vg
        ├── VG_100K
        └── VG_100K_2
```

We are using **LLaVA-332K** for our experiments. You can set the `$TOOLKIT_DIR` environment variable to specify the path to the parent directory of the project root (e.g., `cm/anonymous/toolkitmoe`).

```bash
export TOOLKIT_DIR=./libmoe
deepspeed --include localhost:$ID_GPUS moe_model/train/train_mem.py \
    --deepspeed ./scripts/zero3_offload.json \
    --model_name_or_path microsoft/Phi-3.5-mini-instruct \
    --version phi35 \  
    --data_path $TOOLKIT_DIR/data/jsons/llava_v1_5_mix665k_half.json \
    --image_folder $TOOLKIT_DIR/data \ 
```
