#!/bin/bash
export TMPDIR="/cm/archive/namnv78_new/revise_checkpoints/tmp"
export TOOLKIT_DIR="/cm/archive/namnv78_A100_PDM/LibMoE_Test/vision_language_model"  # Path to the toolkitmoe directory
export KEY_HF="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"       # Hugging Face API key
export ID_GPUS="0,1,2,3"
# Set to -1 to run all steps
export MAX_STEPS=-1
export MODELDIR="phi3mini-clip"
export PYTHONPATH="/cm/archive/namnv78_A100_PDM/LibMoE_Test/vision_language_model":$PYTHONPATH
export TMUX_TMPDIR=~/tmux_tmp
cd /cm/archive/namnv78_A100_PDM/LibMoE_Test/vision_language_model

export LUNA=false
export MODELDIR="phi35-siglip224"
# bash /cm/archive/namnv78_A100_PDM/LibMoE_Test/vision_language_model/scripts/eval/run_eval.sh &
# bash /cm/archive/namnv78_A100_PDM/LibMoE_Test/vision_language_model/imagenet64/train.sh
# bash /cm/archive/namnv78_A100_PDM/LibMoE_Test/vision_language_model/scripts/eval/run_eval1.sh &
# # Vòng lặp để tự động chạy lại nếu có lỗi
export PORT=$(( 20000 + RANDOM % 10000 )) # 20000-29999
export RATE_FLIP=0.07
export ROUTER_COEF=0.001
export THETA=0.1
export WANDB_PROJECT="LibMoE"

export WANDB_API_KEY='ac12c873f730f05f115d320f73a1a908c5db8470'
export NCCL_SOCKET_RETRY_SLEEP_MSEC=1000
export NCCL_IB_TIMEOUT=30
export NCCL_SOCKET_RETRY_CNT=100
export HF_HOME="/cm/shared/namnv78_H102/toolkitmoe/evaluate"
export WANDB_DIR="/cm/archive/namnv78_new/revise_checkpoints/wandb_logs"

export TYPE_MOE="smoe"
export PATH_CHECK_MODEL="/cm/archive/namnv78_new/revise_checkpoints/backup/sft/1M3/Full_smoe/checkpoint-34462"
# export HOME=/cm/archive/namnv78_A100_PDM/home
# bash /cm/archive/namnv78_A100_PDM/LibMoE_Test/vision_language_model/scripts/train/phi35mini/siglip/sparse_upcyling/sft_full1m3.sh
# export TRITON_CACHE_DIR=/cm/archive/namnv78_A100_PDM/triton_cache
# mkdir -p $TRITON_CACHE_DIR
export http_proxy=http://namnv78:nghithoi%4014fpt@10.16.32.11:8080
export https_proxy=http://namnv78:nghithoi%4014fpt@10.16.32.11:8080
# export CUDA_HOME="/cm/archive/namnv78_A100_PDM/home/miniconda3/envs/moe/lib/python3.9/site-packages/nvidia/cuda_runtime"
# source /cm/archive/namnv78_A100_PDM//cm/archive/namnv78_A100_PDM/home/miniconda3/bin/activate 
# conda activate moe

cd /cm/archive/namnv78_A100_PDM/LibMoE_Test/vision_language_model

# echo "Staring stage pretrain"
# bash ./scripts/train/phi35mini/clip/pretrain.sh
# tmux capture-pane -pS - > ./assets/result_eval/phi35mini_clip_pretrain.txt

# echo "Staring stage pft"
# bash ./scripts/train/phi35mini/clip/pft.sh
# tmux capture-pane -pS - > ./assets/result_eval/phi35mini_clip_pft.txt

echo "Staring stage sft"
bash /cm/archive/namnv78_A100_PDM/LibMoE_Test/vision_language_model/scripts/train/phi35mini/siglip/sft.sh
