#!/bin/bash
export TMPDIR="/cm/archive/namnv78/tmp"
export TOOLKIT_DIR="/cm/archive/namnv78"  # Path to the toolkitmoe directory
export KEY_HF=""       # Hugging Face API key
export ID_GPUS="1"
# Set to -1 to run all steps
export MAX_STEPS=-1
export MODELDIR="phi3mini-clip"
export PYTHONPATH="/cm/archive/namnv78/libmoe/LibMoE":$PYTHONPATH
export TMUX_TMPDIR=~/tmux_tmp
export TYPE_MOE="smoe"

# echo "Staring stage pretrain"
# bash ./scripts/train/phi35mini/clip/pretrain.sh
# tmux capture-pane -pS - > ./assets/result_eval/phi35mini_clip_pretrain.txt

# echo "Staring stage pft"
# bash ./scripts/train/phi35mini/clip/pft.sh
# tmux capture-pane -pS - > ./assets/result_eval/phi35mini_clip_pft.txt

echo "Staring stage sft"
bash ./scripts/train/phi3mini/clip/sft.sh
tmux capture-pane -pS - > ./assets/result_eval/phi3mini_clip_sft_test_$TYPE_MOE.txt
