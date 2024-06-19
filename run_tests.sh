#!/bin/bash

# Arrays for the different arguments
data_names=("DEAP_feat" "DREAMER_feat")
data_paths=("eegain/data/features_matrices/DEAP" "eegain/data/features_matrices/DREAMER")
split_types=("LOTO" "LOSO_fixed")
label_types=("V" "A")
input_sizes=(129 117 688)
num_classes_set= (2, 2, 3)

# Hyperparameter search ranges
dropout_rates=(0.1 0.2 0.3 0.4)
num_layers_set=(2 3 4 5)
hidden_sizes=(128 256 512 1024)

# Common arguments
num_epochs=100
batch_size=8
lr=0.001
weight_decay=0
label_smoothing=0.01
channels=32
log_dir="logs/"

# Iterate through each combination of arguments and hyperparameters
for index in "${!data_names[@]}"; do
  data_name=${data_names[$index]}
  data_path=${data_paths[$index]}
  input_size=${input_sizes[$index]}
  num_classes=${num_classes[$index]}
  
  for split_type in "${split_types[@]}"; do
    for label_type in "${label_types[@]}"; do
      for dropout_rate in "${dropout_rates[@]}"; do
        for num_layers in "${num_layers_set[@]}"; do
          for hidden_size in "${hidden_sizes[@]}"; do

            # Determine the log file name
            overall_log_file="log_${split_type}_${data_name}_${label_type}_dropout${dropout_rate}_layers${num_layers}_hidden${hidden_size}.txt"

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
              --num_layers=${num_layers} \
              --hidden_size=${hidden_size} \
              --overall_log_file=${overall_log_file}"

            # Execute the command
            echo "Executing: $cmd"
            eval $cmd
          done
        done
      done
    done
  done
done
