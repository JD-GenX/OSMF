#!/bin/bash




project_path=$1
input_data_path=$2
save_dir_root=$3

image_save_dir=${save_dir_root}/out_images
mkdir -p ${image_save_dir}

# log_file
log_file=${save_dir_root}/log.txt

base_model_path=${project_path}"/data/weight/majicmixRealistic_v7"
controlnet_model_path=${project_path}"/data/weight/ctr_maji7"

accelerate launch diffuser_inference/inference_sd.py  --base_model_path $base_model_path  --save_path $image_save_dir \
          --controlnet_model_path $controlnet_model_path --batch_size 48 --num_inference_steps 30 \
          --sampler_name 'Euler a' --data_path $input_data_path --log_file $log_file

chmod 777 ${log_file}
