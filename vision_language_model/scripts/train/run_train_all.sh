#!/bin/bash

# =============================================================================
# LibMoE Training Configuration Script
# =============================================================================
# This script configures environment variables and runs training stages
# for LibMoE vision-language models.
# =============================================================================

# =============================================================================
# System & Path Configuration
# =============================================================================
export TOOLKIT_DIR="./LibMoE/vision_language_model"
export PYTHONPATH="$TOOLKIT_DIR:$PYTHONPATH"
export TMPDIR="./checkpoints/tmp"
export TMUX_TMPDIR=~/tmux_tmp

# =============================================================================
# Hugging Face Configuration
# =============================================================================
export HF_TOKEN=""
export HF_HOME="./LibMoE/vision_language_model/evaluate"

# =============================================================================
# Weights & Biases (wandb) Configuration
# =============================================================================
export WANDB_PROJECT="LibMoE"
export WANDB_API_KEY=''
export WANDB_DIR="./checkpoints/wandb_logs"

# =============================================================================
# GPU & Distributed Training Configuration
# =============================================================================
export ID_GPUS="0,1,2,3"                    # GPU IDs to use for training
export PORT=$(( 20000 + RANDOM % 10000 ))   # Random port between 20000-29999

# NCCL (NVIDIA Collective Communications Library) settings
export NCCL_SOCKET_RETRY_SLEEP_MSEC=1000
export NCCL_IB_TIMEOUT=30
export NCCL_SOCKET_RETRY_CNT=100

# =============================================================================
# Model Configuration
# =============================================================================
export MODELDIR="phi35-siglip224"           # Model architecture identifier
export TYPE_MOE="smoe"                      # MoE type: smoe, xmoe, etc.

# =============================================================================
# Training Execution
# =============================================================================

# Change to working directory
cd ./LibMoE/vision_language_model 

# -----------------------------------------------------------------------------
# Stage 1: Pre-training (Commented out - uncomment to run)
# -----------------------------------------------------------------------------
# echo "Starting stage: Pre-training"
# bash ./scripts/train/phi35mini/clip/pretrain.sh

# -----------------------------------------------------------------------------
# Stage 2: Pre-finetuning (Commented out - uncomment to run)
# -----------------------------------------------------------------------------
# echo "Starting stage: Pre-finetuning"
# bash ./scripts/train/phi35mini/clip/pft.sh

# -----------------------------------------------------------------------------
# Stage 3: Supervised Fine-tuning (Active)
# -----------------------------------------------------------------------------
echo "Starting stage: Supervised Fine-tuning (SFT)"
bash ./scripts/train/phi35mini/siglip/sft.sh

echo "Training completed!"
