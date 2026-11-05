#!/bin/bash

export PYTHONPATH=$PYTHONPATH:"./"


epoch=$1
ref_model_name_or_path=$2
model_name_or_path=$3
dpo_data_train_set=$4
save_checkpoint_path_epoch_lora=$5
save_checkpoint_path_epoch_merged=$6



echo "epoch=${epoch} ref_model_name_or_path: ${ref_model_name_or_path}"
echo "epoch=${epoch} model_name_or_path: ${model_name_or_path}"
echo "epoch=${epoch} dpo_data_train_set: ${dpo_data_train_set}"
echo "epoch=${epoch} save_checkpoint_path_epoch_lora: ${save_checkpoint_path_epoch_lora}"
echo "epoch=${epoch} save_checkpoint_path_epoch_merged: ${save_checkpoint_path_epoch_merged}"
echo "epoch=$epoch, train ing..."

crop_scale="0,0.20"
drop_scale=0.5
mdpo_weight=1.0
mdpo_beta=0.5
learning_rate=2e-5


per_device_train_batch_size=8
gradient_accumulation_steps=2
save_steps=300
num_train_epochs=5

deepspeed src/shell/train_pm_rm/train_mdpo_reward_append_user.py \
      --deepspeed scripts/zero3.json \
      --model_name_or_path $model_name_or_path \
      --mdpo_weight $mdpo_weight \
      --ref_model_name_or_path $ref_model_name_or_path \
      --version "v1.6" \
      --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 0 \
      --data_path $dpo_data_train_set \
      --vision_tower "path_to_clip-vit-large-patch14-336" \
      --mm_projector_type mlp2x_gelu \
      --mm_vision_select_layer -2 \
      --mm_vision_select_feature "patch"\
      --mm_use_im_start_end False \
      --mm_use_im_patch_token False \
      --bf16 True \
      --output_dir $save_checkpoint_path_epoch_lora \
      --image_aspect_ratio pad \
      --group_by_modality_length False \
      --num_train_epochs 1 \
      --per_device_train_batch_size $per_device_train_batch_size \
      --per_device_eval_batch_size 4 \
      --gradient_accumulation_steps $gradient_accumulation_steps \
      --evaluation_strategy "no" \
      --save_strategy "steps" \
      --save_steps $save_steps \
      --save_total_limit 5 \
      --gamma 0.1    \
      --learning_rate $learning_rate \
      --weight_decay 0. \
      --warmup_ratio 0. \
      --lr_scheduler_type "cosine" \
      --logging_steps 1 \
      --tf32 True \
      --model_max_length 2048 \
      --gradient_checkpointing True \
      --dataloader_num_workers 4 \
      --lazy_preprocess True \
      --report_to tensorboard \
      --crop_scale $crop_scale \
      --drop_scale $drop_scale \
      --mdpo_beta $mdpo_beta


python src/shell/train_pm_rm/merge_lora_weights_user.py --model-path $save_checkpoint_path_epoch_lora --model-base $ref_model_name_or_path --save-model-path $save_checkpoint_path_epoch_merged
