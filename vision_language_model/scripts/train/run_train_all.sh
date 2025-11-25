#!/bin/bash

### -------------------------------
### Environment Setup for LibMoE VLM
### -------------------------------

# Temporary directory for intermediate files
export TMPDIR="/LibMoE/checkpoints/tmp"

# Toolkit path (codebase for VLM training & evaluation)
export TOOLKIT_DIR="/LibMoE/vision_language_model"

# HuggingFace API key (if needed for model/token download)
export KEY_HF=""

# GPUs to use (comma-separated list)
export ID_GPUS="0,1,2,3"

# Set max training steps; use -1 to run all steps
export MAX_STEPS=-1

# Append the VLM toolkit to PYTHONPATH
export PYTHONPATH="${TOOLKIT_DIR}:${PYTHONPATH}"

# Tmux temp directory (prevents crash on some cluster setups)
export TMUX_TMPDIR=~/tmux_tmp

# Model checkpoint directory (e.g., phi-3.5 pre-initialized with SigLIP-224)
export MODELDIR="phi35-siglip224"

# Random port assignment (20000–29999) for distributed communication
export PORT=$((20000 + RANDOM % 10000))

# Logging: W&B project name and directories
export WANDB_PROJECT="LibMoE"
export WANDB_DIR="/LibMoE/wandb_logs"
export WANDB_API_KEY=""  # Set your W&B API key here if using W&B

# HuggingFace cache dir (can prevent redownloads)
export HF_HOME="/LibMoE/toolkitmoe/evaluate"

# NCCL communication tuning (for multi-GPU stability)
export NCCL_SOCKET_RETRY_SLEEP_MSEC=1000
export NCCL_IB_TIMEOUT=30
export NCCL_SOCKET_RETRY_CNT=100

# Type of Mixture-of-Experts (SMoE, MoE, etc.)
export TYPE_MOE="smoe"

cd /LibMoE/vision_language_model

# echo "Staring stage pretrain"
# bash ./scripts/train/phi35mini/clip/pretrain.sh
# tmux capture-pane -pS - > ./assets/result_eval/phi35mini_clip_pretrain.txt

# echo "Staring stage pft"
# bash ./scripts/train/phi35mini/clip/pft.sh
# tmux capture-pane -pS - > ./assets/result_eval/phi35mini_clip_pft.txt

echo "Staring stage sft"
bash ./scripts/train/phi35mini/siglip/sft.sh
