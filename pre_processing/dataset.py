# /usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import pandas as pd
import numpy as np
import config
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from config import logger,CONTENT,SUBJECT,SUBJECTS,MAX_SEQUENCE_LEN
from pre_processing import word_embedding

def loadData(segs_paths, start=0, sample_num=None):
    '''
    加载原始数据
    '''
    frames = []
    # 计算最小的语料数量
    min_sample_num = 1000000
    for path in segs_paths:
        df = pd.read_csv(path)
        if min_sample_num > df.shape[0]:
            min_sample_num = df.shape[0]
    logger.info("min_sample_num = {}".format(min_sample_num))

    # if sample_num != None and sample_num > min_sample_num:
    #     logger.error("参数 sample_num 大于已有的语料规模.")
    #     raise EnvironmentError("参数 sample_num 大于已有的语料规模.")

    for path in segs_paths:
        df = pd.read_csv(path)
        df[CONTENT] = df.apply(func=lambda x:x[CONTENT][:1000], axis=1)
        logger.info("Reading {}. Total articles: {}.".format(path, df.shape[0]))
        # frames.append(df.sample(n=sample_num))
        if sample_num:
            frames.append(df[start:sample_num])
        else:
            frames.append(df)
        df.dropna()
    dfs = pd.concat(frames)
    dfs[CONTENT] = dfs.apply(func=lambda x:x[CONTENT], axis=1)

    # 统计语料
    logger.info("读取预料统计")
    logger.info(str(dfs[config.SUBJECT].value_counts()).replace('    ', '=').replace('\n', '、'))
    return dfs

def splitData(dfs, test_size=0.25):
    '''
    划分数据集
    '''
    documents = dfs[CONTENT].values      # 取内容列
    labels = dfs[SUBJECT].values         # 去标签列

    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = \
        train_test_split(documents, labels, test_size=test_size, random_state=5)
    logger.info("划分数据集={}, 测试集={}".format(len(x_train),len(x_test)))

    return x_train, x_test, y_train, y_test

def encode_data(x_train, x_test, y_train, y_test, model_instance=None):
    # 词编码
    LABEL_ENCODER = LabelEncoder()
    label_en = LABEL_ENCODER.fit(SUBJECTS)
    for_one_hot = label_en.transform(SUBJECTS).reshape((len(SUBJECTS), 1))  # String -> int
    Y_train = LABEL_ENCODER.transform(y_train).reshape((len(y_train), 1))  # 一位数组n reshape->二维数组(n,1)
    Y_test = LABEL_ENCODER.transform(y_test).reshape((len(y_test), 1))

    # One-Hot 编码
    ONE_HOT_ENCODER = OneHotEncoder(sparse=False, categories='auto')
    ONE_HOT_ENCODER.fit(for_one_hot)
    Y_train = ONE_HOT_ENCODER.transform(Y_train)  # 将整数转为one-hot
    Y_test = ONE_HOT_ENCODER.transform(Y_test)

    logger.info("科目: {}".format(SUBJECTS))
    logger.info("label编码: {}".format(LABEL_ENCODER.transform(SUBJECTS)))
    logger.info("one-hot编码: ")
    for i in ONE_HOT_ENCODER.transform(for_one_hot):
        logger.info("{}".format(i))

    # 词编码
    TOKENIZER = Tokenizer(num_words=None, split=' ')
    TOKENIZER.fit_on_texts(x_train)
    # TOKENIZER.fit_on_texts(x_test) # !!!!不能加进来
    X_train = TOKENIZER.texts_to_sequences(x_train)  # 建立词-索引表->词向量嵌入矩阵->词向量
    X_test = TOKENIZER.texts_to_sequences(x_test)
    logger.info("标签One-Hot编码完成；词编码转换完成，共有 {} 词.".format(len(TOKENIZER.index_word)))

    # 固定长度为 MAX_SEQUENCE_LEN，不足则补 0
    X_train = pad_sequences(X_train, padding='post', maxlen=MAX_SEQUENCE_LEN)
    X_test = pad_sequences(X_test, padding='post', maxlen=MAX_SEQUENCE_LEN)
    logger.info("补齐序列完成.")

    # 赋值
    if model_instance:
        model_instance.TOKENIZER = TOKENIZER
        model_instance.LABEL_ENCODER = LABEL_ENCODER
        model_instance.ONE_HOT_ENCODER = ONE_HOT_ENCODER

    return X_train, X_test, Y_train, Y_test

def create_embedding_matrix(word2vec_path, binary=True, model_instance=None):
    '''
    利用预训练的 word2vec 创建嵌入矩阵
    '''
    logger.info("创建词嵌入矩阵...")
    vocab_size = len(model_instance.TOKENIZER.word_index) + 1  # Adding 1 because of reserved 0 index
    word2vec = word_embedding.load_word2vec(word2vec_path, binary)
    embedding_dim = -1
    if word2vec:
        embedding_dim = word2vec.vector_size    # vector_size -> embedding_dim
    else:
        logger.info("word2vec 为空.")

    EMBEDDING_MATRIX = np.zeros((vocab_size, embedding_dim))
    for word, i in model_instance.TOKENIZER.word_index.items():
        if word in word2vec.vocab:
            EMBEDDING_MATRIX[i] = word2vec.word_vec(word)
    # 赋值
    if model_instance:
        model_instance.EMBEDDING_MATRIX = EMBEDDING_MATRIX

    logger.info("词嵌入矩阵创建完成. Shape ({},{})".format(vocab_size, embedding_dim))

if __name__ == '__main__':
    segs_path = [os.path.join(config.SEG_DIR, "{}.csv".format(i)) for i in SUBJECTS]
    loadData(segs_path, sample_num=500)
