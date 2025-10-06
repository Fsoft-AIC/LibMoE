echo "Training ..."

export MOE_TYPE="moe_layer"
export CUDA_VISIBLE_DEVICES="2,6"
export MASTER_PORT=29525
NUM_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)


deepspeed --master_port $MASTER_PORT --include localhost:$CUDA_VISIBLE_DEVICES run_deepspeed.py \
    /cm/shared/anonymous/moeut_training_code/sweeps/experiments/proposed_method/660M/slimpajama_moe_no_attmoe_660M_standardlb.yaml
