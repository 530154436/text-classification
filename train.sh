#!/usr/bin/env bash
# 训练词向量
#python3 -u train_word2vec.py --sg 0 --size 50 --iter 50 --num_cores 8 > logs/word2vec1.log
#python3 -u train_word2vec.py --sg 0 --size 100 --iter 50 --num_cores 8 > logs/word2vec2.log
#python3 -u train_word2vec.py --sg 0 --size 150 --iter 50 --num_cores 8 > logs/word2vec3.log
#python3 -u train_word2vec.py --sg 1 --size 100 --iter 50 --num_cores 8 > logs/word2vec4.log


# 词向量对比
nohup python3 -u train_lstm.py --model_type lstm --sg 0 --size 100  --iter 50 --lstm_num 150 --lstm_drop 0.2 --epochs 200 > logs/lstm1.log &
nohup python3 -u train_lstm.py --model_type lstm --sg 1 --size 150  --iter 50 --lstm_num 150 --lstm_drop 0.2 --epochs 200 > logs/lstm6.log &
nohup python3 -u train_lstm.py --model_type lstm --sg 1 --size 100  --iter 50 --lstm_num 150 --lstm_drop 0.2 --epochs 200 > logs/lstm5.log &
python3 -u train_lstm.py --model_type lstm --sg 0 --size 150  --iter 50 --lstm_num 150 --lstm_drop 0.2 --epochs 200 > logs/lstm2.log

nohup python3 -u train_lstm.py --model_type lstm --sg 1 --size 200  --iter 50 --lstm_num 150 --lstm_drop 0.2 --epochs 200 > logs/lstm7.log &
nohup python3 -u train_lstm.py --model_type lstm --sg 0 --size 200  --iter 50 --lstm_num 150 --lstm_drop 0.2 --epochs 200 > logs/lstm3.log &
nohup python3 -u train_lstm.py --model_type lstm --sg 1 --size 50  --iter 50 --lstm_num 150 --lstm_drop 0.2 --epochs 200 > logs/lstm4.log &
python3 -u train_lstm.py --model_type lstm --sg 0 --size 50  --iter 50 --lstm_num 150 --lstm_drop 0.2 --epochs 200 > logs/lstm0.log

## lstm 单元数 + 失活率
nohup python3 -u train_lstm.py --model_type lstm --sg 1 --size 100  --iter 50 --lstm_num 50 --lstm_drop 0.1 --epochs 200 > logs/lstm8.log &
nohup python3 -u train_lstm.py --model_type lstm --sg 1 --size 100  --iter 50 --lstm_num 50 --lstm_drop 0.2 --epochs 200 > logs/lstm9.log &
nohup python3 -u train_lstm.py --model_type lstm --sg 1 --size 100  --iter 50 --lstm_num 50 --lstm_drop 0.3 --epochs 200 > logs/lstm10.log &
python3 -u train_lstm.py --model_type lstm --sg 1 --size 100  --iter 50 --lstm_num 50 --lstm_drop 0.4 --epochs 200 > logs/lstm11.log

nohup python3 -u train_lstm.py --model_type lstm --sg 1 --size 100  --iter 50 --lstm_num 100 --lstm_drop 0.1 --epochs 200 > logs/lstm12.log &
nohup python3 -u train_lstm.py --model_type lstm --sg 1 --size 100  --iter 50 --lstm_num 100 --lstm_drop 0.2 --epochs 200 > logs/lstm13.log &
nohup python3 -u train_lstm.py --model_type lstm --sg 1 --size 100  --iter 50 --lstm_num 100 --lstm_drop 0.3 --epochs 200 > logs/lstm14.log &
python3 -u train_lstm.py --model_type lstm --sg 1 --size 100  --iter 50 --lstm_num 100 --lstm_drop 0.4 --epochs 200 > logs/lstm15.log

nohup python3 -u train_lstm.py --model_type lstm --sg 1 --size 100  --iter 50 --lstm_num 150 --lstm_drop 0.1 --epochs 200 > logs/lstm16.log &
#nohup python3 -u train_lstm.py --model_type lstm --sg 1 --size 100  --iter 50 --lstm_num 150 --lstm_drop 0.2 --epochs 200 > logs/lstm17.log &
nohup python3 -u train_lstm.py --model_type lstm --sg 1 --size 100  --iter 50 --lstm_num 150 --lstm_drop 0.3 --epochs 200 > logs/lstm18.log &
python3 -u train_lstm.py --model_type lstm --sg 1 --size 100  --iter 50 --lstm_num 150 --lstm_drop 0.4 --epochs 200 > logs/lstm19.log

nohup python3 -u train_lstm.py --model_type lstm --sg 1 --size 100  --iter 50 --lstm_num 200 --lstm_drop 0.1 --epochs 200 > logs/lstm20.log &
nohup python3 -u train_lstm.py --model_type lstm --sg 1 --size 100  --iter 50 --lstm_num 200 --lstm_drop 0.2 --epochs 200 > logs/lstm21.log &
nohup python3 -u train_lstm.py --model_type lstm --sg 1 --size 100  --iter 50 --lstm_num 200 --lstm_drop 0.3 --epochs 200 > logs/lstm22.log &
python3 -u train_lstm.py --model_type lstm --sg 1 --size 100  --iter 50 --lstm_num 200 --lstm_drop 0.4 --epochs 200 > logs/lstm23.log

nohup python3 -u train_lstm.py --model_type lstm --sg 1 --size 100  --iter 50 --lstm_num 250 --lstm_drop 0.1 --epochs 200 > logs/lstm24.log &
nohup python3 -u train_lstm.py --model_type lstm --sg 1 --size 100  --iter 50 --lstm_num 250 --lstm_drop 0.2 --epochs 200 > logs/lstm25.log &
nohup python3 -u train_lstm.py --model_type lstm --sg 1 --size 100  --iter 50 --lstm_num 250 --lstm_drop 0.3 --epochs 200 > logs/lstm26.log &
python3 -u train_lstm.py --model_type lstm --sg 1 --size 100  --iter 50 --lstm_num 250 --lstm_drop 0.4 --epochs 200 > logs/lstm27.log

nohup python3 -u train_lstm.py --model_type lstm --sg 1 --size 200  --iter 50 --lstm_num 150 --lstm_drop 0.2 --epochs 200 > logs/lstm7.log &
nohup python3 -u train_lstm.py --model_type lstm --sg 0 --size 200  --iter 50 --lstm_num 150 --lstm_drop 0.2 --epochs 200 > logs/lstm3.log &

# tensorboard --logdir ./ --host 0.0.0.0 --port 8889
#usage: tensorboard [-h] [--helpfull] [--logdir PATH] [--host ADDR]
#                   [--port PORT] [--purge_orphaned_data BOOL]
#                   [--reload_interval SECONDS] [--db URI] [--db_import]
#                   [--db_import_use_op] [--inspect] [--tag TAG]
#                   [--event_file PATH] [--path_prefix PATH]
#                   [--window_title TEXT] [--max_reload_threads COUNT]
#                   [--reload_task TYPE]
#                   [--samples_per_plugin SAMPLES_PER_PLUGIN]
#                   [--master_tpu_unsecure_channel ADDR]
#                   [--debugger_data_server_grpc_port PORT]
#                   [--debugger_port PORT]

# 无法显示 tfboard -> https://stackoverflow.com/questions/37987839/how-can-i-run-tensorboard-on-a-remote-server
#1. from your local machine, run
#ssh -N -f -L localhost:16006:localhost:6006 <user@remote>
#ssh -N -f -L localhost:16007:localhost:6007 root@112.90.89.16 -p 10033

#2. on the remote machine, run:
#tensorboard --logdir <path> --port 6006
#tensorboard --logdir logs/ --port 6007 --host 0.0.0.0
#
#Then, navigate to (in this example) http://localhost:16006 on your local machine.
#(explanation of ssh command:
#-N : no remote commands
#-f : put ssh in the background
#-L <machine1>:<portA>:<machine2>:<portB> :
#
#forward <machine2>:<portB> (remote scope) to <machine1>:<portA> (local scope)

#kill -9 `ps aux | grep train_lstm|awk '{print $2}'`

