#!/bin/bash

# Arrays for the different arguments
data_names=("DREAMER_feat" "DEAP_feat")
data_paths=("eegain/data/features_matrices/DREAMER" "eegain/data/features_matrices/DEAP")
split_types=("LOTO" "LOSO_fixed")
label_types=("V" "A")
input_sizes=(129 117)

# Common arguments
num_classes=2
num_epochs=20
batch_size=8
lr=0.001
weight_decay=0
label_smoothing=0.01
dropout_rate=0.2
channels=32
log_dir="logs/"

# Iterate through each combination of arguments
for index in "${!data_names[@]}"; do
  data_name=${data_names[$index]}
  data_path=${data_paths[$index]}
  input_size=${input_sizes[$index]}
  
  for split_type in "${split_types[@]}"; do
    for label_type in "${label_types[@]}"; do

      # Determine the log file name
      overall_log_file="log_${split_type}_${data_name}_${label_type}.txt"

      # Construct the python command
      cmd="python3 run_cli.py \
        --model_name=MLP \
        --data_name=${data_name} \
        --data_path=${data_path} \
        --split_type=${split_type} \
        --num_classes=${num_classes} \
        --label_type=${label_type} \
        --num_epochs=${num_epochs} \
        --batch_size=${batch_size} \
        --lr=${lr} \
        --weight_decay=${weight_decay} \
        --label_smoothing=${label_smoothing} \
        --dropout_rate=${dropout_rate} \
        --channels=${channels} \
        --log_dir=${log_dir} \
        --input_size=${input_size} \
        --overal_log_file=${overall_log_file}"

      # Execute the command
      echo "Executing: $cmd"
      eval $cmd
    done
  done
done
