#!/bin/bash




model_path=$1
input_data_path=$2
output_data_path=$3
epoch_save_root=$4
patch_size=${5:-2}

prompt_nums=2

rm -rf ${epoch_save_root}/sd_prompt_split_*.json
rm -rf ${epoch_save_root}/input_split_*.json
echo "${epoch_save_root}/sd_prompt_split_*.json"
#sleep 3m
python src/shell/train_pm_rm/split_input_json_and_merge_output.py --tag "split" --patch_size $patch_size --base_data_path $input_data_path --save_path $epoch_save_root


if [ ! -d "$epoch_save_root" ]; then
    echo "Error: directory '$epoch_save_root' does not exist."
    exit 1
fi

# shellcheck disable=SC2207
files=($(find "$epoch_save_root" -name "input_split*.json"))
#
file_count=${#files[@]}

echo "found $file_count matching files."

if [ "$file_count" -gt 0 ]; then
    echo "starting loop..."

    for ((i=0; i<file_count; i++))
    do
        echo "executing round $((i+1)) of $file_count rounds"
        current_file="${files[i]}"
        echo "processing file: $current_file"
        split_output_data_path=${epoch_save_root}'/sd_prompt_split_'${i}'.json'
        echo "will output: $split_output_data_path"

        port=$((i + 29500))
        setsid accelerate launch --main_process_port $port \
        llava/eval/inference_llava_train_pm_user_stage2.py --model-path $model_path --output_data_path $split_output_data_path \
              --generate_nums $prompt_nums --base_data_path $current_file --temperature 1.0 &

        echo "executing prompt processing $current_file"
    done
else
    echo "no matching files found."
    exit 0
fi

sleep_times=0
while true;
do
  # shellcheck disable=SC2207
  sd_prompt_files=($(find "$epoch_save_root" -name "sd_prompt_split*.json"))
  sd_prompt_file_count=${#sd_prompt_files[@]}
  if [ "$sd_prompt_file_count" -eq "$file_count" ]; then
      echo "sleep 30S to prevent files just generated"
      sleep 30
      echo "input file count $file_count, sd_prompt output count $sd_prompt_file_count equal"
      echo "$epoch_save_root merging sd_prompt_split files"

      python src/shell/train_pm_rm/split_input_json_and_merge_output.py --tag "merge" --patch_size $patch_size --base_data_path $input_data_path --save_path $epoch_save_root
      #
      if [ -e "$output_data_path" ]; then
          echo "Both $output_data_path exist."
          echo "$epoch_save_root SD output normal"
          chmod 777 ${output_data_path}
      else
          echo "current $epoch_save_root an exception occurred, SD Prompt output abnormal, needs to be retrained"
      fi
      exit 0
  else
      echo "input file count $file_count, sd_prompt output count $sd_prompt_file_count, not equal"
      echo "SD Prompt generating...waiting 2 minutes, wait count $sleep_times"
      sleep 2m

      if [ "$sleep_times" -eq "2000" ]; then
          echo "SD output may have an exception"
          echo "$epoch_save_root current epoch training incomplete, SD output may have an exception"
          exit 0
      else

          sleep_times=$((sleep_times + 1))
          if pgrep -f "llava/eval/inference_llava_train_pm_user_stage2.py" > /dev/null
          then
              echo "llava/eval/inference_llava_train_pm_user_stage2.py is running"
              echo "still generating SDPrompt..."
          else
              echo "llava/eval/inference_llava_train_pm_user_stage2.py is not running"
              sleep_times=$((sleep_times + 50))
          fi
      fi
  fi
done

chmod 777 ${output_data_path}
