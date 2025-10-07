 
#!/bin/bash
export API_KEY='.....'
export KEY_HF="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"
# export CUDA_LAUNCH_BLOCKING=0
export HF_HOME="/cm/archive/anonymous/checkpoints/benchmarks"
export TMPDIR="/cm/shared/anonymous_H102/tmp"
export TOOLKIT_DIR="/cm/shared/anonymous_H102"  # Path to the toolkitmoe directory
export CUDA_LAUNCH_BLOCKING=0
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

export CUDA_VISIBLE_DEVICES="4,6"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
# curl http://0.0.0.0:8080/v1/models
python /cm/shared/anonymous_H102/toolkitmoe/evaluate/analysis/server.py