# !python run_client.py --model_name="TSception"
## example model names : EEGNet, TSception, DeepConvNet, ShallowConvNet, RANDOM
## example data names: DEAP, MAHNOB, SEED_IV, AMIGOS, DREAMER
## set channel and n_chan to 14 for amigos and dreamer and 32 for mahnob and deap, SEED_IV - 62 
python run_cli.py \
--model_name=MLP \
--data_name=SEED_IV_feat \
--data_path='/home/akarasmanoglou/Documents/Github/EEGain/book/datasets/features_matrices/SEED IV' \
--split_type="LOTO" \
--num_classes=3 \
--sampling_r=128 \
--window=4 \
--label_type="V" \
--num_epochs=200 \
--batch_size=16 \
--lr=0.001 \
--weight_decay=0 \
--label_smoothing=0.01 \
--dropout_rate=0.5 \
--channels=32 \
--log_dir="logs/" \
--input_size=691 \
--overal_log_file="log_file_name.txt" \