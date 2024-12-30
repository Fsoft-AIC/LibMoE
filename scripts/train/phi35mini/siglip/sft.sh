#!/bin/bash 
checkpoints="/cm/archive/namnv78"


deepspeed --master_port 29520 --include localhost:$ID_GPUS moe_model/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $checkpoints/checkpoints/$MODELDIR/pft \
    --version phi35 \
    --data_path $checkpoints/data/jsons/llava_v1_5_mix665k_half.json \
    --image_folder $checkpoints/data \
    --vision_tower google/siglip-so400m-patch14-224 \
    --vision_tower_dir $checkpoints/checkpoints/$MODELDIR/pft/clip.bin \
    --scales 1,3 \
    --pretrain_mm_mlp_adapter $checkpoints/checkpoints/$MODELDIR/pft/mm_projector.bin \
    --mm_projector_type moe \
    --mlp_smoe true \
    --clip_smoe true \
    --moe_name $TYPE_MOE \
    --num_experts 4 \
    --num_selected 2 \
    --sparse_upcycling true \
    --balance_loss_coef 0.01 \
    --router_z_loss_coef 0.001 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $checkpoints/checkpoints/$MODELDIR/sft/$TYPE_MOE \
    --num_train_epochs 1 \
    --per_device_train_batch_size 5 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1664 \
    --save_total_limit 16 \
    --learning_rate 4e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --max_steps $MAX_STEPS
