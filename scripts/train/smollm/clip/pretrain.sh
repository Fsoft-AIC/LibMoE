#!/bin/bash


deepspeed --include localhost:$ID_GPUS moe_model/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path HuggingFaceTB/SmolLM-1.7B-Instruct \
    --version smollm \
    --data_path $TOOLKIT_DIR/data/llava/llava_pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder $TOOLKIT_DIR/data/llava/llava_pretrain/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --vision_tower_dir $TOOLKIT_DIR/checkpoints/clip-vit-large-patch14-336/pytorch_model.bin \
    --scales 1,3 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $TOOLKIT_DIR/checkpoints/$MODELDIR/pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
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
    --cache_dir $TOOLKIT_DIR/checkpoints/smollm-1.7b-instruct \
    --max_steps $MAX_STEPS
