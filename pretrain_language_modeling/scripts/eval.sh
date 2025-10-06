echo "Evaluation ..."
# source /cm/archive/anonymous/miniconda3/bin/activate moeut


export MOE_TYPE="moe_layer"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export MASTER_PORT=29545
NUM_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)


### ================ EVAL ALL ===================
# export CHECKPOINT_DIR="/cm/shared/anonymous/moeut_training_code/save/slimpajama_moe_no_attmoe_154M_standard_lb/checkpoint"
# file_list=$(find "$CHECKPOINT_DIR" -type f)

# for file_path in $file_list; do
#     echo "Processing file: $file_path"
#     torchrun --master_port $MASTER_PORT --nproc_per_node=$NUM_DEVICES paper/moe_universal/run_tests.py \
#         --path_weight "$file_path" \
#         --save_dir /cm/shared/anonymous/moeut_training_code/save/slimpajama_moe_no_attmoe_154M_standard_lb/tmp \
#         --bs 16
# done
### ==============================================


### =============== EVAL SINGLE ==================
torchrun --master_port $MASTER_PORT --nproc_per_node=$NUM_DEVICES paper/moe_universal/run_tests.py \
    --path_weight /cm/shared/anonymous/moeut_training_code/save/slimpajama_moe_no_attmoe_154M_standard_lb/checkpoint/model-90000.pth \
    --save_dir /cm/shared/anonymous/moeut_training_code/save/slimpajama_moe_no_attmoe_154M_standard_lb/tmp \
    --bs 16
### ==============================================

# python3 paper/moe_universal/run_tests.py \
#     --path_weight /cm/shared/anonymous/moeut_training_code/save/slimpajama_moe_no_attmoe_154M_standard_lb/checkpoint/model-100000.pth \
#     --save_dir /cm/shared/anonymous/moeut_training_code/tmp \
#     --bs 16