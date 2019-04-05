#!/usr/bin/env bash
python3 -u main.py --sg 0 --size 100 --iter 50 --lstm_num 100 --lstm_drop 0.2 --dense_num 100 --epochs 200 > lstm1.log
python3 -u main.py --sg 0 --size 200 --iter 50 --lstm_num 100 --lstm_drop 0.2 --dense_num 100 --epochs 200 > lstm2.log
python3 -u main.py --sg 1 --size 100 --iter 50 --lstm_num 100 --lstm_drop 0.2 --dense_num 100 --epochs 200 > lstm3.log
python3 -u main.py --sg 0 --size 200 --iter 50 --lstm_num 50 --lstm_drop 0.2 --dense_num 100 --epochs 200 > lstm4.log
python3 -u main.py --sg 0 --size 200 --iter 50 --lstm_num 150 --lstm_drop 0.2 --dense_num 100 --epochs 200 > lstm5.log