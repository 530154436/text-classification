#!/usr/bin/env bash
# 训练词向量

# 词向量对比
#python3 -u train_lstm.py --model_type lstm --sg 0 --size 50 --iter 50 --lstm_num 100 --lstm_drop 0.2 --dense_num 100 --epochs 200 > logs/lstm0.log
#python3 -u train_lstm.py --model_type lstm --sg 0 --size 100 --iter 50 --lstm_num 100 --lstm_drop 0.2 --dense_num 100 --epochs 200 > logs/lstm1.log
#python3 -u train_lstm.py --model_type lstm --sg 0 --size 150 --iter 50 --lstm_num 100 --lstm_drop 0.2 --dense_num 100 --epochs 200 > logs/lstm8.log
#python3 -u train_lstm.py --model_type lstm --sg 0 --size 200 --iter 50 --lstm_num 100 --lstm_drop 0.2 --dense_num 100 --epochs 200 > logs/lstm3.log
#python3 -u train_lstm.py --model_type lstm --sg 1 --size 100 --iter 50 --lstm_num 100 --lstm_drop 0.2 --dense_num 100 --epochs 200 > logs/lstm2.log

## lstm 单元数
#python3 -u train_lstm.py --model_type lstm --sg 0 --size 150 --iter 50 --lstm_num 50 --lstm_drop 0.2 --dense_num 100 --epochs 200 > logs/lstm12.log
#python3 -u train_lstm.py --model_type lstm --sg 0 --size 150 --iter 50 --lstm_num 150 --lstm_drop 0.2 --dense_num 100 --epochs 200 > logs/lstm4.log
#python3 -u train_lstm.py --model_type lstm --sg 0 --size 150 --iter 50 --lstm_num 200 --lstm_drop 0.2 --dense_num 100 --epochs 200 > logs/lstm5.log
#python3 -u train_lstm.py --model_type lstm --sg 0 --size 150 --iter 50 --lstm_num 250 --lstm_drop 0.2 --dense_num 100 --epochs 200 > logs/lstm9.log

# lstm 失活率
#python3 -u train_lstm.py --model_type lstm --sg 0 --size 150 --iter 50 --lstm_num 150 --lstm_drop 0.4 --dense_num 100 --epochs 200 > logs/lstm6.log
#python3 -u train_lstm.py --model_type lstm --sg 0 --size 150 --iter 50 --lstm_num 150 --lstm_drop 0.0 --dense_num 100 --epochs 200 > logs/lstm7.log
#python3 -u train_lstm.py --model_type lstm --sg 0 --size 150 --iter 50 --lstm_num 150 --lstm_drop 0.3 --dense_num 100 --epochs 200 > logs/lstm10.log
#python3 -u train_lstm.py --model_type lstm --sg 0 --size 150 --iter 50 --lstm_num 150 --lstm_drop 0.1 --dense_num 100 --epochs 200 > logs/lstm11.log

# dense 单元数
python3 -u train_lstm.py --model_type lstm --sg 0 --size 150 --iter 50 --lstm_num 150 --lstm_drop 0.2 --dense_num 50 --epochs 200 > logs/lstm13.log
python3 -u train_lstm.py --model_type lstm --sg 0 --size 150 --iter 50 --lstm_num 150 --lstm_drop 0.2 --dense_num 150 --epochs 200 > logs/lstm14.log
python3 -u train_lstm.py --model_type lstm --sg 0 --size 150 --iter 50 --lstm_num 150 --lstm_drop 0.2 --dense_num 200 --epochs 200 > logs/lstm15.log
python3 -u train_lstm.py --model_type lstm --sg 0 --size 150 --iter 50 --lstm_num 150 --lstm_drop 0.2 --dense_num 250 --epochs 200 > logs/lstm16.log


# 不同神经网络模型
#python3 -u train_lstm.py --model_type bilstm --sg 0 --size 200 --iter 50 --lstm_num 100 --lstm_drop 0.2 --dense_num 100 --epochs 200 > logs/lstm6.log