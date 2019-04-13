# /usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import argparse
import tensorflow as tf
from keras.callbacks import  ModelCheckpoint
from keras import backend as K
from keras.callbacks import TensorBoard
from model.LSTM import TextRNN_LSTM
from pre_processing.dataset import loadData, splitData, encode_data, create_embedding_matrix
from config import SG,SIZE,ITER,NUM_CORES,BATCH_SIZE,MAX_SEQUENCE_LEN,LSTM_NUM,LSTM_DROP,DENSE_NUM,EPOCHS, \
                   SEG_DIR, MODEL_DIR, CLASS_NUM, SUBJECTS

import config

####################################################################
# cpu 核数设置
lstm_config = tf.ConfigProto(intra_op_parallelism_threads=NUM_CORES,
                        inter_op_parallelism_threads=NUM_CORES,
                        allow_soft_placement=True,
                        device_count = {'CPU' : NUM_CORES})
session = tf.Session(config=lstm_config)
K.set_session(session)
####################################################################

def train(model_type, segs_path, word2vec_path):
    text_rnn = TextRNN_LSTM(class_num=CLASS_NUM)

    # 1. 加载数据集
    dfs = loadData(segs_path,sample_num=config.SAMPLE_NUM)

    # 2. 划分数据集
    x_train, x_test, y_train, y_test = splitData(dfs)

    # 3. 对数据集和标签进行编码
    X_train, X_test, Y_train, Y_test = encode_data(x_train, x_test, y_train, y_test, model_instance=text_rnn)

    # 4. 创建嵌入矩阵
    create_embedding_matrix(word2vec_path, binary=True, model_instance=text_rnn)

    # 5. 训练神经网络
    # 监听最优模型
    bst_model_path = os.path.join(
        MODEL_DIR,
        'vsg{}.vs{}.vi{}.ln{}.dn{}.ld{}.{}.check_point'.format(SG, SIZE, ITER, LSTM_NUM, DENSE_NUM, LSTM_DROP, model_type))
    # 当监测值不再改善时，该回调函数将中止训练
    # early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    # 该回调函数将在每个epoch后保存模型到 bst_model_path
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=False)

    # 可视化
    tb = TensorBoard(log_dir='./logs',      # graph 目录
                     histogram_freq=1,       # 按照何等频率（epoch）来计算直方图，0为不计算
                     batch_size=BATCH_SIZE,  # 用多大量的数据计算直方图
                     write_graph=False,      # 是否存储网络结构图
                     write_grads=False,      # 是否可视化梯度直方图
                     write_images=False,     # 是否可视化参数
                     embeddings_freq=0,
                     embeddings_layer_names=None,
                     embeddings_metadata=None)

    # 网络结构、目标函数设置
    text_rnn.set_model(max_sequence_length=MAX_SEQUENCE_LEN,
                      lstm_drop=LSTM_DROP,
                      lstm_num=LSTM_NUM,
                      dense_num=DENSE_NUM,
                      last_activation='softmax',
                      loss='categorical_crossentropy',
                      metrics=('accuracy', 'categorical_accuracy')
                    )
    text_rnn.train(X_train,Y_train,X_test,Y_test, BATCH_SIZE, EPOCHS, model_checkpoint,tb)

def main(model_type):
    # 设定路径
    segs_path = [os.path.join(SEG_DIR, "{}.csv".format(i)) for i in SUBJECTS]
    word2vec_path = os.path.join(MODEL_DIR, "vector.sg{}.size{}.iter{}.bin".format(SG, SIZE, ITER))
    train(model_type, segs_path, word2vec_path)

def parse():
    '''
    解析命令行参数
    '''
    parser = argparse.ArgumentParser()
    # word2vec
    parser.add_argument("--sg", dest='SG',   required=True, type=int)
    parser.add_argument("--size", dest='SIZE', required=True, type=int)
    parser.add_argument("--iter", dest='ITER', required=True, type=int )

    parser.add_argument("--num_cores", dest='NUM_CORES', required=False, type=int)
    parser.add_argument("--window", dest='WINDOW', required=False, type=int)
    parser.add_argument("--min_count", dest='MIN_COUNT', required=False, type=int)

    # lstm
    parser.add_argument("--lstm_num", dest='LSTM_NUM', required=True, type=int )
    parser.add_argument("--lstm_drop", dest='LSTM_DROP', required=True, type=float )
    parser.add_argument("--dense_num", dest='DENSE_NUM', required=True, type=int)
    parser.add_argument("--epochs", dest='EPOCHS', required=True, type=int)
    parser.add_argument("--model_type", dest='model_type', required=True, type=str)

    # 语料数量
    parser.add_argument("--sample_num", dest='SAMPLE_NUM', required=False, type=int)

    args = parser.parse_args()
    config.SG = args.SG
    config.SIZE = args.SIZE
    config.ITER = args.ITER
    config.LSTM_NUM = args.LSTM_NUM
    config.LSTM_DROP = args.LSTM_DROP
    config.DENSE_NUM = args.DENSE_NUM
    config.EPOCHS = args.EPOCHS
    model_type = args.model_type

    if args.NUM_CORES:
        config.NUM_CORES = args.NUM_CORES
    if args.WINDOW:
        config.WINDOW = args.WINDOW
    if args.MIN_COUNT:
        config.MIN_COUNT = args.MIN_COUNT
    if args.SAMPLE_NUM:
        config.SAMPLE_NUM = args.SAMPLE_NUM

    return model_type

if __name__ == '__main__':
    model_type = parse()
    main(model_type)
