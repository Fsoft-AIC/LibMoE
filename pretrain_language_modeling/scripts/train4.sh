echo "Training ..."

export MOE_TYPE="moe_layer_deepseek"
export CUDA_VISIBLE_DEVICES="4,5,6,7"
export MASTER_PORT=29548
NUM_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)


torchrun --master_port $MASTER_PORT --nproc_per_node=$NUM_DEVICES run.py  \
    /cm/archive/anonymous/moeut_training_code/sweeps/experiments/proposed_method/660M/slimpajama_moe_no_attmoe_660M_standardlb_deepseek_shared_only.yaml