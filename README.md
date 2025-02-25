# VLM_Safety_Transfer

This is the code for our paper Cross-Modal Safety Mechanism Transfer in Large Vision-Language Models (ICLR 2025)
### Analyze the misalignment phenomenon of hidden states in VLM
```
sh ./tox_analysis/tox_llava.sh
sh ./tox_analysis/tox_llava_no_image.sh
sh ./tox_analysis/read_analysis_llava.sh
```

###  Preatrain
```bash
#!/bin/bash

deepspeed ./TGA/train_TGA.py \
    --deepspeed ./TGA/scripts/zero3_fp32.json \
    --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2  \
    --version llava_llama_2 \
    --train_fusion False \
    --data_path /path_to_lava_v1_5_mix665k.json \
    --image_folder /path_to_images \
    --vision_tower clip-vit-large-patch14-336  \
    --pretrain_mm_mlp_adapter /path_to_projector \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /path_to_output \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 1 \
    --learning_rate 2e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1200 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True

```

### Fine-Tune

```bash
#!/bin/bash

deepspeed ./TGA/train_TGA.py \
    --deepspeed ./TGA/scripts/zero3_fp32.json \
    --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2  \
    --version llava_llama_2 \
    --train_fusion False \
    --data_path /path_to_lava_v1_5_mix665k.json \
    --image_folder /path_to_images \
    --vision_tower clip-vit-large-patch14-336  \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /path_to_output \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 1 \
    --learning_rate 2e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1200 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True

```
