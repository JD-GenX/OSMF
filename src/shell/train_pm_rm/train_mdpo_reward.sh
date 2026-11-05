#!/bin/bash


echo "CPU info:"
cat /proc/cpuinfo | grep "model name" | uniq
echo "CPU cores: $(nproc)"
echo "CPU usage: $(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1"%"}')"

echo -e "\nMemory info:"
free -h

echo -e "\nDisk usage:"
df -h

echo -e "\nSystem load:"
uptime

if command -v nvidia-smi &> /dev/null
then
    echo -e "\nGPU info:"
    nvidia-smi
fi

unset LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:"./"



epoch=${1:-0}
train_id=${2:-'prompt'}
is_cover=${3:-0}
MODEL_VERSION='v1.6-vicuna-7b'
project_path='your_project_path'
reward_model_path='path_to_reward_model'
train_data_path='path_to_train_data_path'
ref_model_name_or_path="path_to_pretrained_G-MLLM"

save_root=${project_path}/data/train_PM_RM/${train_id}
mkdir -p ${save_root}
chmod 777 ${save_root}
epoch_save_root=${save_root}/"epoch_${epoch}"
mkdir -p ${epoch_save_root}
rm -rf $epoch_save_root/_SUCCESS
mkdir -p ${epoch_save_root}/run
chmod 777 ${epoch_save_root}
chmod 777 ${epoch_save_root}/run

# --------------- start step1 ------------------------
start_time=$(date +%s)
start_time_formatted=$(date +"%Y-%m-%d %H:%M:%S")
echo -e "epoch=${epoch} Prompt generation start time: $start_time_formatted"



if [ $epoch -eq 0 ]; then
    model_name_or_path=$ref_model_name_or_path
else
  before_epoch=$((epoch-1))
  model_name_or_path=${save_root}/"epoch_${before_epoch}"/llava-$MODEL_VERSION-finetune-dpo-epoch-${before_epoch}-merged
fi


input_data_path=${train_data_path}
output_data_path=${epoch_save_root}'/sd_prompt.json'

# shellcheck disable=SC1073
if [ $is_cover -eq 0 ] && [ -e "$output_data_path" ]; then
    echo -e "already generated before, and not overwriting output (is_cover 0)"
else
    bash src/shell/train_pm_rm/1-train_mdpo_reward_pool.sh "${model_name_or_path}" "${input_data_path}" "${output_data_path}" "${epoch_save_root}"
fi

chmod 777 ${output_data_path}

end_time=$(date +%s)
end_time_formatted=$(date +"%Y-%m-%d %H:%M:%S")
execution_time=$((end_time - start_time))
echo -e "epoch=${epoch} Prompt generation start time: $start_time_formatted\nPrompt generation end time: $end_time_formatted\nExecution time: $((execution_time / 3600))h $(((execution_time % 3600) / 60))m $((execution_time % 60))s"
# --------------- end step1 ------------------------


# --------------- start step2 ------------------------
start_time=$(date +%s)
start_time_formatted=$(date +"%Y-%m-%d %H:%M:%S")
echo -e "epoch=${epoch} Background image generation start time: $start_time_formatted"

if [ $is_cover -eq 0 ] && [ -e "$output_data_path" ] && [ ! -e "${epoch_save_root}/log.txt" ]; then
    echo "SD prompt exists: ${output_data_path} exist"
    bash src/shell/train_pm_rm/2-train_mdpo_reward.sh "${project_path}" "${output_data_path}" "${epoch_save_root}"
fi

end_time=$(date +%s)
end_time_formatted=$(date +"%Y-%m-%d %H:%M:%S")
execution_time=$((end_time - start_time))
echo -e "epoch=${epoch} Background image generation start time: $start_time_formatted\nBackground image generation end time: $end_time_formatted\nExecution time: $((execution_time / 3600))h $(((execution_time % 3600) / 60))m $((execution_time % 60))s"
# --------------- end step2 ------------------------


## --------------- start step3 ------------------------
start_time=$(date +%s)
start_time_formatted=$(date +"%Y-%m-%d %H:%M:%S")
echo -e "epoch=${epoch} Reward scoring start time: $start_time_formatted"
reward_output_data_path_source=${epoch_save_root}'/reward.json'
reward_output_data_path=${epoch_save_root}'/reward_out.json'

if [ $is_cover -eq 0 ] && [ -e "$reward_output_data_path" ]; then
    echo -e "reward_out already generated before, and not overwriting output (is_cover 0)"
    echo -e "reward_out already generated before, and not overwriting output (is_cover 0)"
else
    bash src/shell/train_pm_rm/3-train_mdpo_reward.sh "${reward_model_path}" "${epoch_save_root}" "${reward_output_data_path}"
fi

chmod 777 ${reward_output_data_path_source}
chmod 777 ${reward_output_data_path}

end_time=$(date +%s)
end_time_formatted=$(date +"%Y-%m-%d %H:%M:%S")
execution_time=$((end_time - start_time))
echo -e "epoch=${epoch} Reward scoring start time: $start_time_formatted\nReward scoring end time: $end_time_formatted\nExecution time: $((execution_time / 3600))h $(((execution_time % 3600) / 60))m $((execution_time % 60))s"
## --------------- end step3 ------------------------
#


## --------------- start step4 ------------------------
start_time=$(date +%s)
start_time_formatted=$(date +"%Y-%m-%d %H:%M:%S")
echo -e "epoch=${epoch} Prompt Model optimization with positive/negative samples start time: $start_time_formatted"
MODEL_VERSION="v1.6-vicuna-7b"

if [ $epoch -eq 0 ]; then
    model_name_or_path=$ref_model_name_or_path
else
  before_epoch=$((epoch-1))
  model_name_or_path=${save_root}/"epoch_${before_epoch}"/llava-$MODEL_VERSION-finetune-dpo-epoch-${before_epoch}-merged
fi

save_checkpoint_path_epoch_lora=$epoch_save_root/llava-$MODEL_VERSION-finetune-dpo-epoch-${epoch}-lora
save_checkpoint_path_epoch_merged=$epoch_save_root/llava-$MODEL_VERSION-finetune-dpo-epoch-${epoch}-merged
dpo_data_train_set=${reward_output_data_path}

if [ -e "$dpo_data_train_set" ]; then
    echo "dpo_data_train_set exists: ${dpo_data_train_set} exist"
    bash src/shell/train_pm_rm/4-train_mdpo_reward.sh "${epoch}" "${ref_model_name_or_path}" "${model_name_or_path}" "${dpo_data_train_set}" "${save_checkpoint_path_epoch_lora}" "${save_checkpoint_path_epoch_merged}" | tee $epoch_save_root/train_mdpo_reward_${epoch}.log
fi

echo "epoch=${epoch} ref_model_name_or_path: ${ref_model_name_or_path}"
echo "epoch=${epoch} model_name_or_path: ${model_name_or_path}"
echo "epoch=${epoch} dpo_data_train_set: ${dpo_data_train_set}"
echo "epoch=${epoch} save_checkpoint_path_epoch_merged: ${save_checkpoint_path_epoch_merged}"
echo "epoch=$epoch"

chmod 777 $save_checkpoint_path_epoch_lora
chmod 777 $save_checkpoint_path_epoch_merged


end_time=$(date +%s)
end_time_formatted=$(date +"%Y-%m-%d %H:%M:%S")
execution_time=$((end_time - start_time))
echo -e "epoch=${epoch} Prompt Model optimization with positive/negative samples start time: $start_time_formatted\nPrompt Model optimization with positive/negative samples end time: $end_time_formatted\nExecution time: $((execution_time / 3600))h $(((execution_time % 3600) / 60))m $((execution_time % 60))s"
## --------------- end step4 ------------------------
#

if [ -e "$output_data_path" ] && [ -e "${epoch_save_root}/log.txt" ] && [ -e "$reward_output_data_path" ] && [ -e "$save_checkpoint_path_epoch_merged" ]; then
    echo "Both $output_data_path exist."
    echo "Both ${epoch_save_root}/log.txt exist."
    echo "Both $reward_output_data_path exist."
    echo "Both $save_checkpoint_path_epoch_merged exist."
    echo "current epoch=$epoch training normal"
    touch "$epoch_save_root"/_SUCCESS
else
    echo "current epoch=$epoch training incomplete, an exception occurred, needs to be retrained"
fi


rm -r ${epoch_save_root}/run

