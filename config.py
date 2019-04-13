# /usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import logging
import numpy as np

BASE_DIR = "/Users/zhengchubin/Desktop/text_classification/"
# BASE_DIR = "/home/ai/text_classification/"

CORPUS_DIR = os.path.join(BASE_DIR, 'corpus')
SEG_DIR = os.path.join(BASE_DIR, 'seg')
MODEL_DIR = os.path.join(BASE_DIR, 'model')

if not os.path.exists(CORPUS_DIR): os.mkdir(CORPUS_DIR)
if not os.path.exists(SEG_DIR): os.mkdir(SEG_DIR)
if not os.path.exists(MODEL_DIR): os.mkdir(MODEL_DIR)

GRADE = '年级'
SUBJECT = '科目'
TITLE = '标题'
CONTENT = '内容'
DOC_URL = 'URL'

# 日志
logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)

SUBJECTS=np.array(["语文", "数学", "英语", "物理", "化学", "生物", "历史", "地理", "政治"])
CLASS_NUM = len(SUBJECTS)

################################
#  模型参数设置
################################
# word2vec
SG = 0                                               # 算法模型 0:"cbow";1:"skg"
SIZE = 100                                           # 词向量维度
ITER = 50                                            # 迭代次数
WINDOW = 5                                           # 窗口
MIN_COUNT = 5                                        # 最小词频
################################
# 神经网络
LSTM = 'lstm'
BI_LSTM = 'bilstm'
MODEL_TYPES = [LSTM, BI_LSTM]
MAX_SEQUENCE_LEN = 1000                              # 序列最长长度
BATCH_SIZE = 32                                      # 批大小
EPOCHS = 5                                           # 迭代次数
LSTM_DROP = 0.2                                      # LSTM 丢掉率
LSTM_NUM = 100                                       # LSTM 单元数
DENSE_NUM = 100                                      # DENSE 单元数

# 训练参数
NUM_CORES = 8
################################
# 模型保存
TOKENIZER = None
LABEL_ENCODER = None
ONE_HOT_ENCODER = None
EMBEDDING_MATRIX = None
RNN_MODEL = None