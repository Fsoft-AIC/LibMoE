echo "Training ..."

export MOE_TYPE="moe_layer"
export CUDA_VISIBLE_DEVICES="4,5,6,7"
export MASTER_PORT=29536
NUM_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)


# Run the torchrun command and check for errors
torchrun --master_port $MASTER_PORT --nproc_per_node=$NUM_DEVICES run.py \
    /cm/shared/anonymous/moeut_training_code/sweeps/experiments/proposed_method/test.yaml