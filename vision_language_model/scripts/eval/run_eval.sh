#!/bin/bash
export API_TYPE=''                          # OpenAI API type
export DEPLOYMENT=''                        # OpenAI API deployment
export ENDPOINT=''                          # OpenAI API endpoint
export VERSION=''                           # OpenAI API version
export API_KEY=''                           # OpenAI API key
export KEY_HF=""                            # Hugging Face API key
export CUDA_LAUNCH_BLOCKING=0
export HF_HOME="../evaluate/lmms-eval"         # Path to lmms-eval
export TMPDIR=""                            # Path to tmp directory
export TOOLKIT_DIR=""                       # Path to the toolkitmoe directory
export PYTHONPATH="../LibMoE":$PYTHONPATH


# GPU setup
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# List of models to evaluate
models=(
    # conv_template|pretrained_path
    "phi3|/checkpoints/phi3mini-siglip224/sft/smoe"  
)

# Evaluation
cd /LibMoE/vision_language_model/evaluate
for model in "${models[@]}"
do
    part=$(echo $model | cut -d'|' -f1)
    PRETRAINED_PATH=$(echo $model | cut -d'|' -f2)
    
    # Task list
    echo "Running evaluation for model: $PRETRAINED_PATH with conv_template: $part"
    tasks=(
        "ai2d"
        "textvqa"
        "gqa" 
        "mmbench_en" 
        "mme" 
        "mmmu_val" 
        "mmstar" 
        "pope" 
        "scienceqa" 
        "mathvista_testmini" 
        "hallusion_bench_image"
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
            --output_path ./logs/ 
    done
done
