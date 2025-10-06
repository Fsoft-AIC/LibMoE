echo "Training ..."

export MOE_TYPE="moe_layer"
export CUDA_VISIBLE_DEVICES="4,5,6,7"
export MASTER_PORT=29547
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO
NUM_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)


torchrun --master_port $MASTER_PORT --nproc_per_node=$NUM_DEVICES run.py  \
    /cm/archive/anonymous/moeut_training_code/sweeps/experiments/proposed_method/660M/slimpajama_moe_no_attmoe_660M_sigmoid_standardlb.yaml


# bash /cm/shared/anonymous/moeut_training_code/scripts/train4.sh
