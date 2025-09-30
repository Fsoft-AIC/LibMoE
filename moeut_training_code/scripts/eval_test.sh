echo "Evaluation ... "

# GPU setup
export MOE_TYPE="moe_layer"
export CUDA_VISIBLE_DEVICES="5,6"
export PYTHONPATH="/cm/shared/anonymous/moeut_training_code:$PYTHONPATH"
export MASTER_PORT=29551
NUM_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# prepare tests
tasks=(
    # all tasks in the framework/dataset/text
    "lambada"
    "cbt"
    "hellaswag"
    "piqa"
    "blimp"
    "ai2arc"
    # additional tasks
    "mmlu"
    "openbookqa"
    "winogrande"
    "siqa"
    "commonsenseqa"
    "race"
)

# run tests
checkpoint_path="/cm/shared/anonymous/moeut_training_code/save/slimpajama_moe_no_attmoe_660M_standardlb/checkpoint"
save_dir="/cm/shared/anonymous/moeut_training_code/save/slimpajama_moe_no_attmoe_660M_standardlb/export"
bs=16

for task in "${tasks[@]}"; do
    tasks_script="${tasks_script} -lm.eval.${task}.enabled 1"
done

echo $tasks_script

# check if checkpoint_path is a file or directory
if [ -f "$checkpoint_path" ]; then
    echo "Checkpoint path is a file"
    # run tests single
    # torchrun --master_port $MASTER_PORT --nproc_per_node=$NUM_DEVICES paper/moe_universal/run_tests.py \
    #     --tasks "$tasks_script" \
    #     --path_weight "$checkpoint_path" \
    #     --save_dir "$save_dir" \
    #     --bs "$bs"
    deepspeed --master_port $MASTER_PORT --include localhost:$CUDA_VISIBLE_DEVICES paper/moe_universal/run_tests_deepspeed.py \
        --tasks "$tasks_script" \
        --path_weight "$checkpoint_path" \
        --save_dir "$save_dir" \
        --bs "$bs"
else
    echo "Checkpoint path is a directory"
    # run tests all
    file_list=$(find "$checkpoint_path" -type f)
    for file_path in $file_list; do
        echo "Processing file: $file_path"
        # torchrun --master_port $MASTER_PORT --nproc_per_node=$NUM_DEVICES paper/moe_universal/run_tests.py \
        #     --tasks "$tasks_script" \
        #     --path_weight "$file_path" \
        #     --save_dir "$save_dir" \
        #     --bs "$bs"
        deepspeed --master_port $MASTER_PORT --include localhost:$CUDA_VISIBLE_DEVICES paper/moe_universal/run_tests_deepspeed.py \
            --tasks "$tasks_script" \
            --path_weight "$file_path" \
            --save_dir "$save_dir" \
            --bs "$bs"
    done
fi

echo "Evaluation done ..."

# bash /cm/shared/anonymous/moeut_training_code/scripts/eval_test2.sh