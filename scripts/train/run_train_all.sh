#!/bin/bash
export TMPDIR=""
export TOOLKIT_DIR="."  # Path to the toolkitmoe directory
export KEY_HF=""       # Hugging Face API key
# export KEY_HF=""       # Hugging Face API key
export ID_GPUS="2,3"
# Set to -1 to run all steps
export MAX_STEPS=-1
export MODELDIR="phi35-siglip224"
export PYTHONPATH="./LibMoE":$PYTHONPATH
export TMUX_TMPDIR=~/tmux_tmp
export TYPE_MOE="smoe"

# echo "Staring stage pretrain"
# bash ./scripts/train/phi35mini/clip/pretrain.sh
# tmux capture-pane -pS - > ./assets/result_eval/phi35mini_clip_pretrain.txt

# echo "Staring stage pft"
# bash ./scripts/train/phi35mini/clip/pft.sh
# tmux capture-pane -pS - > ./assets/result_eval/phi35mini_clip_pft.txt

echo "Staring stage sft"
bash ./LibMoE/scripts/train/phi35mini/siglip/sft.sh
tmux capture-pane -pS - > ./assets/result_eval/phi3mini_clip_sft_test_$TYPE_MOE.txt
