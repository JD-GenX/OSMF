#!/bin/bash



reward_model_checkpoint_path=$1
save_dir_root=$2
out_data_path=$3

input_data_path=${save_dir_root}'/sd_prompt.json'
log_file=${save_dir_root}/log.txt

accelerate launch llava/eval/eval_cluster_pm_rm.py  --model-path $reward_model_checkpoint_path --output_data_path $out_data_path --temperature 0.0 --base_data_path $input_data_path --log_file_path $log_file


