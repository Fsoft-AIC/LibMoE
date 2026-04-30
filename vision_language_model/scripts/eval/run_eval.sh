#!/bin/bash

# =============================================================================
# Azure OpenAI Configuration:
# Use these settings only if evaluating with Azure OpenAI API.
# For OpenAI API usage, leave these blank and ensure that
# code in tasks (see ./vision_language_model/evaluate/lmms_eval/tasks) is set up for OpenAI instead.
# You may need to modify task code to properly support Azure endpoints.
# =============================================================================
export AZURE_OPENAI_API_TYPE=''             # Azure OpenAI API type (e.g., 'azure')
export AZURE_OPENAI_DEPLOYMENT=''           # Azure deployment name
export AZURE_OPENAI_ENDPOINT=''             # Azure endpoint URL
export AZURE_OPENAI_API_VERSION=''          # Azure API version (e.g., '2023-05-15')
export AZURE_OPENAI_API_KEY=''              # Azure OpenAI API key

# =============================================================================
# OpenAI API Configuration (for tasks requiring standard OpenAI API)
# Don't use this configuration if you want to use Azure OpenAI API
# =============================================================================
export OPENAI_API_KEY=''                    # OpenAI API key (from platform.openai.com)
export OPENAI_ORG_ID=''                     # OpenAI organization ID (optional)

# =============================================================================
# Hugging Face Configuration
# =============================================================================
export HF_TOKEN=""                          # Hugging Face API token
export HF_HOME="./checkpoints/benchmarks"

# =============================================================================
# System Configuration
# =============================================================================
export CUDA_LAUNCH_BLOCKING=0
export TMPDIR="./checkpoints/tmp"
export TOOLKIT_DIR="./LibMoE/vision_language_model"
export PYTHONPATH="$TOOLKIT_DIR:$PYTHONPATH"


# GPU setup
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
export CUDA_VISIBLE_DEVICES="0"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# List of models to evaluate
models=(
    # conv_template|pretrained_path
    "phi35|./checkpoints/Xphi35-siglip224/SMOE/665K36/Full_smoe"  
)

# Evaluation
cd ./vision_language_model/evaluate
for model in "${models[@]}"
do
    part=$(echo $model | cut -d'|' -f1)
    PRETRAINED_PATH=$(echo $model | cut -d'|' -f2)
    
    # Task list
    echo "Running evaluation for model: $PRETRAINED_PATH with conv_template: $part"
    tasks=(
        "mmmu_val,mmbench_en" # you can add more tasks here and split by comma
    )

    # Run evaluation
    for task in "${tasks[@]}"
    do
        task_name=$(echo $task | tr ',' '_')
        python3 -m accelerate.commands.launch \
            --main_process_port=29511 \
            --num_processes=$num_gpus \
            -m lmms_eval \
            --model llava \
            --model_args pretrained="$PRETRAINED_PATH",conv_template="$part" \
            --tasks $task \
            --batch_size 1 \
            --log_samples \
            --log_samples_suffix llava_v1.5_${task_name} \
            --output_path $PRETRAINED_PATH/logs/ \
            # --return_id_experts True # if you want to log information to analyst behavior of the SMoE
    
    done
done
