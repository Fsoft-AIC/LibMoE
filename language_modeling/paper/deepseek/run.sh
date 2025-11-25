#!/bin/bash

python3 expert_coactivation.py \
    --logits_folder /home/fpt/moeut_training_code/paper/deepseek/logits/679M/smoe_shared \
    --top_k 8 \
    --save_folder /home/fpt/moeut_training_code/paper/deepseek/expert_coactivation
