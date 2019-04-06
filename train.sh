#!/usr/bin/env bash
# 词向量对比
#python3 -u main.py --model_type lstm --sg 0 --size 100 --iter 50 --lstm_num 100 --lstm_drop 0.2 --dense_num 100 --epochs 200 > lstm1.log
python3 -u main.py --model_type lstm --sg 0 --size 200 --iter 50 --lstm_num 100 --lstm_drop 0.2 --dense_num 100 --epochs 200 > lstm2.log
#python3 -u main.py --model_type lstm --sg 1 --size 100 --iter 50 --lstm_num 100 --lstm_drop 0.2 --dense_num 100 --epochs 200 > lstm3.log

# lstm 参数
#python3 -u main.py --model_type lstm --sg 0 --size 200 --iter 50 --lstm_num 50 --lstm_drop 0.2 --dense_num 100 --epochs 200 > lstm4.log
#python3 -u main.py --model_type lstm --sg 0 --size 200 --iter 50 --lstm_num 150 --lstm_drop 0.2 --dense_num 100 --epochs 200 > lstm5.log

# 不同神经网络模型
#python3 -u main.py --model_type bilstm --sg 0 --size 200 --iter 50 --lstm_num 100 --lstm_drop 0.2 --dense_num 100 --epochs 200 > lstm6.log