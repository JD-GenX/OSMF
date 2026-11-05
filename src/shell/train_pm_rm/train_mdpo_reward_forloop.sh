#!/bin/bash

pip install accelerate==0.34.2
# pip install transformers==4.37.2

start_time_forloop=$(date +%s)
start_time_forloop_formatted=$(date +"%Y-%m-%d %H:%M:%S")
echo "mDPO training start time: $start_time_forloop_formatted"


is_cover=${1:-0}

num_train_epochs=20

for (( i=0; i<num_train_epochs; i++ ))
do
  train_id='pm_rm_first'
  save_root=data/train_PM_RM/${train_id}
  epoch_save_root=${save_root}/"epoch_${i}"

  if [ $is_cover -eq 0 ]; then
      if [ -e "$epoch_save_root/_SUCCESS" ]; then
          echo "Both $epoch_save_root/_SUCCESS exist."
      else
          echo "$epoch_save_root/_SUCCESS do not exist"
          bash src/shell/train_pm_rm/train_mdpo_reward.sh $i $train_id

          if [ -e "$epoch_save_root/_SUCCESS" ]; then
            echo "Both $epoch_save_root/_SUCCESS exist."
          else
            echo "need to retry because training is abnormal, $epoch_save_root/_SUCCESS do not exist."
            exit 0
          fi
      fi
      continue
  else
      rm -rf ${epoch_save_root}
      bash src/shell/train_pm_rm/train_mdpo_reward.sh $i $train_id
      if [ -e "$epoch_save_root/_SUCCESS" ]; then
          echo "Both $epoch_save_root/_SUCCESS exist."
      else
          echo "$epoch_save_root/_SUCCESS do not exist"
          bash src/shell/train_pm_rm/train_mdpo_reward.sh $i $train_id

          if [ -e "$epoch_save_root/_SUCCESS" ]; then
            echo "Both $epoch_save_root/_SUCCESS exist."
          else
            echo "need to retry because training is abnormal, $epoch_save_root/_SUCCESS do not exist."
            exit 0
          fi
      fi
  fi

done

end_time_forloop=$(date +%s)
end_time_forloop_formatted=$(date +"%Y-%m-%d %H:%M:%S")
execution_time=$((end_time_forloop - start_time_forloop))

echo -e "mDPO training start time: $start_time_forloop_formatted\n mDPO training end time: $end_time_forloop_formatted\nExecution time: $((execution_time / 3600))h $(((execution_time % 3600) / 60))m $((execution_time % 60))s"
