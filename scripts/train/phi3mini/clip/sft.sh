#!/bin/bash 

deepspeed --include localhost:$ID_GPUS --master_port 60001 moe_model/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $TOOLKIT_DIR/checkpoints/$MODELDIR/pft \
    --version phi3 \
    --data_path $TOOLKIT_DIR/data/jsons/llava_v1_5_mix665k_half.json \
    --image_folder $TOOLKIT_DIR/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --vision_tower_dir $TOOLKIT_DIR/checkpoints/$MODELDIR/pft/clip.bin \
    --scales 1,3 \
    --pretrain_mm_mlp_adapter $TOOLKIT_DIR/checkpoints/$MODELDIR/pft/mm_projector.bin \
    --mm_projector_type moe \
    --mlp_smoe true \
    --clip_smoe true \
    --moe_name $TYPE_MOE \
    --num_experts 4 \
    --num_selected 2 \
    --balance_loss_coef 0.01 \
    --router_z_loss_coef 0.001 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $TOOLKIT_DIR/$MODELDIR/sft/test1_$TYPE_MOE \
    --num_train_epochs 1 \
    --per_device_train_batch_size 5 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 832 \
    --save_total_limit 13 \
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
