echo "Training ..."

export MOE_TYPE="moe_layer"
export CUDA_VISIBLE_DEVICES="6,7"
export MASTER_PORT=29548
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO

NUM_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
cd /cm/archive/namnv78_A100_PDM/libmoe_release/language_modeling

torchrun --master_port $MASTER_PORT --nproc_per_node=$NUM_DEVICES run.py  \
    /cm/archive/namnv78_A100_PDM/libmoe_release/language_modeling/sweeps/154M/slimpajama_moe_no_attmoe_154M_standard_lb.yaml


# bash /cm/shared/anonymous/moeut_training_code/scripts/train4.sh
