# coding=utf-8
import os
import argparse
import pickle
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras import models
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.callbacks import  ModelCheckpoint
from data_collecting.common import logger,MODEL_DIR,SEG_DIR
from data_collecting import common
from pre_processing import word_embedding
from model.TextRNN import TextRNN
import tensorflow as tf
from keras import backend as K

SUBJECT=np.array(["语文", "数学", "英语", "物理", "化学", "生物", "历史", "地理", "政治"])
CLASS_NUM = len(SUBJECT)

################################
#  模型参数设置
################################
# word2vec
SG = 0                                               # 算法模型 0:"cbow";1:"skg"
SIZE = 300                                           # 词向量维度
ITER = 50                                            # 迭代次数
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
################################
# cpu 核数设置
NUM_CORES = 8
config = tf.ConfigProto(intra_op_parallelism_threads=NUM_CORES,
                        inter_op_parallelism_threads=NUM_CORES,
                        allow_soft_placement=True,
                        device_count = {'CPU' : NUM_CORES})
session = tf.Session(config=config)
K.set_session(session)
################################

# 模型保存
TOKENIZER = None
LABEL_ENCODER = None
ONE_HOT_ENCODER = None
EMBEDDING_MATRIX = None
RNN_MODEL = None

def loadData(segs_paths, sample_num=195):
    '''
    加载原始数据
    '''
    frames = []
    for path in segs_paths:
        df = pd.read_csv(path)
        logger.info("Reading {}. Total articles: {}.".format(path, df.shape[0]))
        # frames.append(df.sample(n=sample_num))
        frames.append(df[:sample_num])
        df.dropna()
    dfs = pd.concat(frames)
    return dfs

def preprocess(dfs):
    '''
    预处理，对原始数据进行编码、交叉校验划分、补齐序列
    '''
    global LABEL_ENCODER, ONE_HOT_ENCODER, TOKENIZER

    documents = dfs[common.CONTENT].values      # 取内容列
    labels = dfs[common.SUBJECT].values         # 去标签列

    # 划分训练集和测试集
    documents_train, documents_test, y_train, y_test = \
        train_test_split(documents, labels, test_size=0.1, random_state=1000)

    logger.info("划分数据集={}, 测试集={}".format(len(documents_train),len(documents_test)))

    # 对标签进行编码
    LABEL_ENCODER = LabelEncoder()
    label_en = LABEL_ENCODER.fit(SUBJECT)
    for_one_hot = label_en.transform(SUBJECT).reshape((len(SUBJECT),1))     # String -> int
    Y_train = LABEL_ENCODER.transform(y_train).reshape((len(y_train), 1))   # 一位数组n reshape->二维数组(n,1)
    Y_test = LABEL_ENCODER.transform(y_test).reshape((len(y_test), 1))

    ONE_HOT_ENCODER = OneHotEncoder(sparse=False, categories='auto')
    ONE_HOT_ENCODER.fit(for_one_hot)
    Y_train = ONE_HOT_ENCODER.transform(Y_train)                            # 将整数转为one-hot
    Y_test = ONE_HOT_ENCODER.transform(Y_test)

    logger.info("科目: {}".format(SUBJECT))
    logger.info("label编码: {}".format(LABEL_ENCODER.transform(SUBJECT)))
    logger.info("one-hot编码: ")
    for i in ONE_HOT_ENCODER.transform(for_one_hot):
        logger.info("{}".format(i))

    # 对每个词进行编码
    TOKENIZER = Tokenizer(num_words=None, split=' ')
    TOKENIZER.fit_on_texts(documents)
    X_train = TOKENIZER.texts_to_sequences(documents_train)  # 建立词-索引表->词向量嵌入矩阵->词向量
    X_test = TOKENIZER.texts_to_sequences(documents_test)
    logger.info("标签One-Hot编码完成；词编码转换完成，共有 {} 词.".format(len(TOKENIZER.index_word)))

    # 固定长度为 MAX_SEQUENCE_LEN，不足则补 0
    X_train = pad_sequences(X_train, padding='post', maxlen=MAX_SEQUENCE_LEN)
    X_test = pad_sequences(X_test, padding='post', maxlen=MAX_SEQUENCE_LEN)
    logger.info("补齐序列完成.")

    return X_train,X_test,Y_train,Y_test

def create_embedding_matrix(word2vec_path, binary=True):
    '''
    利用预训练的 word2vec 创建嵌入矩阵
    '''
    global EMBEDDING_MATRIX

    logger.info("创建词嵌入矩阵...")
    vocab_size = len(TOKENIZER.word_index) + 1  # Adding 1 because of reserved 0 index
    word2vec = word_embedding.load_word2vec(word2vec_path, binary)
    embedding_dim = -1
    if word2vec:
        embedding_dim = word2vec.vector_size    # vector_size -> embedding_dim
    else:
        logger.info("word2vec 为空.")

    EMBEDDING_MATRIX = np.zeros((vocab_size, embedding_dim))
    for word, i in TOKENIZER.word_index.items():
        if word in word2vec.vocab:
            EMBEDDING_MATRIX[i] = word2vec.word_vec(word)
    logger.info("词嵌入矩阵创建完成. Shape ({},{})".format(vocab_size, embedding_dim))

def train(model_type, segs_path, word2vec_path):
    global EMBEDDING_MATRIX,RNN_MODEL
    # 前期预处理
    dfs = loadData(segs_path)
    X_train, X_test, Y_train, Y_test = preprocess(dfs)
    create_embedding_matrix(word2vec_path, binary=True)

    # 监听最优模型
    bst_model_path = os.path.join(
        MODEL_DIR,
        'vsg{}.vs{}.vi{}.ln{}.dn{}.ld{}.{}.check_point'.format(SG, SIZE, ITER, LSTM_NUM, DENSE_NUM, LSTM_DROP, model_type))

    # 当监测值不再改善时，该回调函数将中止训练
    # early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    # 该回调函数将在每个epoch后保存模型到 bst_model_path
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=False)

    if model_type == LSTM:
        # 网络结构、目标函数设置
        text_rnn = TextRNN(EMBEDDING_MATRIX, class_num=CLASS_NUM)
        RNN_MODEL = text_rnn.get_model(MAX_SEQUENCE_LEN,
                                       lstm_drop=LSTM_DROP,
                                       lstm_num=LSTM_NUM,
                                       dense_num=DENSE_NUM,
                                       last_activation='softmax',
                                       loss='categorical_crossentropy',
                                       metrics=('accuracy', 'categorical_accuracy')
                                       )

        RNN_MODEL.fit(X_train, Y_train,
              batch_size=BATCH_SIZE,                            # 随机梯度下降批大小
              epochs=EPOCHS,                                    # 迭代次数
              shuffle=True,                                     # 是否打乱数据集
              validation_data=(X_test, Y_test),                 # 验证集
              callbacks=[model_checkpoint])
    elif model_type == BI_LSTM:
        pass

    save_model(model_type)

def save_model(model_type):
    '''
    保存模型
    '''
    global LABEL_ENCODER, ONE_HOT_ENCODER, TOKENIZER, MBEDDING_MATRIX, RNN_MODEL

    # 保存 LabelEncoder、OneHotEncoder、Tokenizer、embedding_matrix
    with open(os.path.join(MODEL_DIR, 'label_encoder.pickle'), 'wb') as f:
        pickle.dump(LABEL_ENCODER, f)
    logger.info("{} 已保存.".format(os.path.join(MODEL_DIR, 'label_encoder.pickle')))

    with open(os.path.join(MODEL_DIR, 'one_hot_encoder.pickle'), 'wb') as f:
        pickle.dump(ONE_HOT_ENCODER, f)
    logger.info("{} 已保存.".format(os.path.join(MODEL_DIR, 'one_hot_encoder.pickle')))

    with open(os.path.join(MODEL_DIR, 'text_tokenizer.pickle'), 'wb') as f:
        pickle.dump(TOKENIZER, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("{} 已保存.".format(os.path.join(MODEL_DIR, 'text_tokenizer.pickle')))

    with open(os.path.join(MODEL_DIR, 'embedding_matrix.pickle'), 'wb') as f:
        pickle.dump(EMBEDDING_MATRIX, f)
    logger.info("{} 已保存.".format(os.path.join(MODEL_DIR, 'embedding_matrix.pickle')))

    RNN_MODEL.save(os.path.join(
        MODEL_DIR, 'vsg{}.vs{}.vi{}.ln{}.dn{}.ld{}.{}.h5' .format(SG, SIZE, ITER, LSTM_NUM, DENSE_NUM, LSTM_DROP, model_type)))
    logger.info("{} 已保存.".format(
        os.path.join(MODEL_DIR, 'vsg{}.vs{}.vi{}.ln{}.dn{}.ld{}.{}.h5' .format(SG, SIZE, ITER, LSTM_NUM, DENSE_NUM, LSTM_DROP, model_type))))

def load_model(model_type):
    '''
    加载模型
    '''
    global LABEL_ENCODER, ONE_HOT_ENCODER, TOKENIZER, EMBEDDING_MATRIX, RNN_MODEL

    # 读取 LabelEncoder、OneHotEncoder、Tokenizer、embedding_matrix
    with open(os.path.join(MODEL_DIR, 'label_encoder.pickle'), 'rb') as f:
        LABEL_ENCODER = pickle.load(f)
    logger.info("{} 已加载.".format(os.path.join(MODEL_DIR, 'label_encoder.pickle')))

    with open(os.path.join(MODEL_DIR, 'one_hot_encoder.pickle'), 'rb') as f:
        ONE_HOT_ENCODER = pickle.load(f)
    logger.info("{} 已加载.".format(os.path.join(MODEL_DIR, 'one_hot_encoder.pickle')))

    with open(os.path.join(MODEL_DIR, 'text_tokenizer.pickle'), 'rb') as f:
        TOKENIZER = pickle.load(f)
    logger.info("{} 已加载.".format(os.path.join(MODEL_DIR, 'text_tokenizer.pickle')))

    with open(os.path.join(MODEL_DIR, 'embedding_matrix.pickle'), 'rb') as f:
        EMBEDDING_MATRIX = pickle.load(f)
    logger.info("{} 已加载.".format(os.path.join(MODEL_DIR, 'embedding_matrix.pickle')))

    RNN_MODEL = models.load_model(
        os.path.join(MODEL_DIR, 'vsg{}.vs{}.vi{}.ln{}.dn{}.ld{}.{}.h5' .format(SG, SIZE, ITER, LSTM_NUM, DENSE_NUM, LSTM_DROP, model_type)))
    logger.info("{} 已加载.".format(
        os.path.join(MODEL_DIR, 'vsg{}.vs{}.vi{}.ln{}.dn{}.ld{}.{}.h5' .format(SG, SIZE, ITER, LSTM_NUM, DENSE_NUM, LSTM_DROP, model_type))))

def predict_documents(documents, model_type=LSTM):
    '''
    预测
    :param documents:  ['a b c', 'c d f']
    :param model_type:
    :param model_dir:
    :return:
    '''
    global LABEL_ENCODER,ONE_HOT_ENCODER, TOKENIZER, MBEDDING_MATRIX, RNN_MODEL
    new_y = []
    x = TOKENIZER.texts_to_sequences(documents)
    x = pad_sequences(x, padding='post', maxlen=MAX_SEQUENCE_LEN)
    y = RNN_MODEL.predict(x, batch_size=BATCH_SIZE)
    for row in y :
        new_y.append((row == row.max(axis=0)) + 0)
    y = ONE_HOT_ENCODER.inverse_transform(new_y)
    y = LABEL_ENCODER.inverse_transform(y.ravel())
    return y

def main(model_type):
    # 设定路径
    segs_path = [os.path.join(SEG_DIR, "{}.csv".format(i)) for i in SUBJECT]
    word2vec_path = os.path.join(MODEL_DIR, "vector.sg{}.size{}.iter{}.bin".format(SG, SIZE, ITER))
    train(model_type, segs_path, word2vec_path)

    # load_model(LSTM, model_dir)
    # y = predict_documents(['魏晋 五言诗 三首 设计 示例 学习 魏晋 五言诗 体例 认识到 五言诗 我国 古典 诗歌 史上', '牵牛星 思想 内容 艺术 特色'])
    # print(y)


def parse():
    '''
    解析命令行参数
    '''
    global SG,SIZE,ITER,LSTM_NUM,LSTM_DROP,DENSE_NUM,EPOCHS
    parser = argparse.ArgumentParser()
    # word2vec
    parser.add_argument("--sg", dest='SG',   required=True, type=int)
    parser.add_argument("--size", dest='SIZE', required=True, type=int)
    parser.add_argument("--iter", dest='ITER', required=True, type=int )

    # lstm
    parser.add_argument("--lstm_num", dest='LSTM_NUM', required=True, type=int )
    parser.add_argument("--lstm_drop", dest='LSTM_DROP', required=True, type=float )
    parser.add_argument("--dense_num", dest='DENSE_NUM', required=True, type=int)
    parser.add_argument("--epochs", dest='EPOCHS', required=True, type=int)
    parser.add_argument("--model_type", dest='model_type', required=True, type=str)
    args = parser.parse_args()
    SG = args.SG
    SIZE = args.SIZE
    ITER = args.ITER
    LSTM_NUM = args.LSTM_NUM
    LSTM_DROP = args.LSTM_DROP
    DENSE_NUM = args.DENSE_NUM
    EPOCHS = args.EPOCHS
    model_type = args.model_type
    return model_type

if __name__ == '__main__':
    model_type = parse()
    main(model_type)
    print(EPOCHS)