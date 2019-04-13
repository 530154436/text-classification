# /usr/bin/env python3
# -*- coding:utf-8 -*-

from keras import Sequential
from keras.layers import LSTM,Dense,Embedding,Dropout
from keras.optimizers import Adam
from model.RNN import RNN
import os
import config
import pickle
from sklearn import metrics
from pre_processing.dataset import loadData,splitData
from keras.preprocessing.sequence import pad_sequences
from keras import models
from config import MODEL_DIR,SEG_DIR,logger

save_log_pattern = '{} 已保存.'
load_log_pattern = '{} 已加载.'

label_save_path = os.path.join(MODEL_DIR, 'label_encoder.pickle')
one_hot_save_path = os.path.join(MODEL_DIR, 'one_hot_encoder.pickle')
tokenizer_save_path = os.path.join(MODEL_DIR, 'text_tokenizer.pickle')
embedding_matrix_path = os.path.join(MODEL_DIR, 'embedding_matrix.pickle')
lstm_save_path = os.path.join(MODEL_DIR,'vsg{}.vs{}.vi{}.ln{}.dn{}.ld{}.{}.h5'.format(
    config.SG, config.SIZE, config.ITER, config.LSTM_NUM, config.DENSE_NUM, config.LSTM_DROP, config.LSTM))

if config.SAMPLE_NUM:
    label_save_path = os.path.join(MODEL_DIR, 'label_encoder.sn{}.pickle'.format(config.SAMPLE_NUM))
    one_hot_save_path = os.path.join(MODEL_DIR, 'one_hot_encoder.sn{}.pickle'.format(config.SAMPLE_NUM))
    tokenizer_save_path = os.path.join(MODEL_DIR, 'text_tokenizer.sn{}.pickle'.format(config.SAMPLE_NUM))
    embedding_matrix_path = os.path.join(MODEL_DIR, 'embedding_matrix.sn{}.pickle'.format(config.SAMPLE_NUM))
    lstm_save_path = os.path.join(MODEL_DIR, 'vsg{}.vs{}.vi{}.ln{}.dn{}.ld{}.{}.sn{}.h5'.format(
        config.SG, config.SIZE, config.ITER, config.LSTM_NUM, config.DENSE_NUM, config.LSTM_DROP, config.LSTM,config.SAMPLE_NUM))

class TextRNN_LSTM(RNN):
    def __init__(self,
                 class_num=9,
                 tokenizer=None,
                 label_encoder=None,
                 one_hot_encoder=None,
                 embedding_matrix=None,
                 rnn_model=None
                 ):
        super(TextRNN_LSTM, self).__init__(class_num)
        # 模型保存
        self.TOKENIZER = tokenizer
        self.LABEL_ENCODER = label_encoder
        self.ONE_HOT_ENCODER = one_hot_encoder
        self.EMBEDDING_MATRIX = embedding_matrix
        self.RNN_MODEL = rnn_model

    def set_model(self,
                  max_sequence_length,
                  lstm_drop=0.2,
                  lstm_num=200,
                  dense_num=200,
                  last_activation='softmax',
                  loss='categorical_crossentropy',
                  metrics = ('accuracy', 'categorical_accuracy')
                  ):
        # embedding_layer = self.get_keras_embedding()
        max_features,embedding_dim  = self.EMBEDDING_MATRIX.shape
        embedding_layer = Embedding(max_features,  # embedding layer
                                    embedding_dim,
                                    weights=[self.EMBEDDING_MATRIX],
                                    input_length=max_sequence_length,
                                    trainable=False)
        model = Sequential()
        model.add(embedding_layer)
        model.add(LSTM(lstm_num,                                        # LSTM layer
                       dropout=lstm_drop))
        model.add(Dropout(lstm_drop))
        model.add(Dense(dense_num,                                      # Dense layer
                        activation='relu'))
        model.add(Dropout(lstm_drop))
        model.add(Dense(self.class_num,
                        activation=last_activation))                    # Dense layer

        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(loss=loss,
                      optimizer=adam,
                      metrics=[item for item in metrics],
                      )
        model.summary()
        self.RNN_MODEL = model
        # plot_model(model, to_file='pic/lstm.png')
        return model

    def train(self, X_train, Y_train,X_test,Y_test, BATCH_SIZE, EPOCHS, model_checkpoint,tb):
        self.RNN_MODEL.fit(X_train, Y_train,
                      batch_size=BATCH_SIZE,  # 随机梯度下降批大小
                      epochs=EPOCHS,  # 迭代次数
                      shuffle=True,  # 是否打乱数据集
                      validation_data=(X_test, Y_test),  # 验证集
                      callbacks=[model_checkpoint, tb])
        self.save_model()

    def metric(self):
        '''
        评估
        '''
        # 加载数据
        dfs = loadData([os.path.join(SEG_DIR, "{}.csv".format(i)) for i in config.SUBJECTS])
        # 划分训练集和测试集
        x_train, x_test, y_train, y_test_labels = splitData(dfs, test_size=0.25)

        SUBJECTS = config.SUBJECTS.tolist()
        subject_index = {}
        for i, sub in enumerate(SUBJECTS):
            subject_index[sub] = i
        y_true = [subject_index[y] for y in y_test_labels]

        # 加载模型
        self.load_model()
        y_predict_labels = self.predict_documents(x_test)
        y_predict = [subject_index[y] for y in y_predict_labels]
        report = metrics.classification_report(y_true, y_predict, target_names=SUBJECTS)
        print(report)
        print("平均准确率\n", metrics.accuracy_score(y_true, y_predict))
        print("混淆矩阵\n", metrics.confusion_matrix(y_test_labels, y_predict_labels, labels=SUBJECTS))

    def predict_documents(self, x):
        '''
        预测
        :param x:  ['a b c', 'c d f']
        '''
        new_y = []
        x = self.TOKENIZER.texts_to_sequences(x)
        x = pad_sequences(x, padding='post', maxlen=config.MAX_SEQUENCE_LEN)
        y = self.RNN_MODEL.predict(x, batch_size=config.BATCH_SIZE)
        for row in y:
            new_y.append((row == row.max(axis=0)) + 0)
        y = self.ONE_HOT_ENCODER.inverse_transform(new_y)
        y = self.LABEL_ENCODER.inverse_transform(y.ravel())
        return y

    def load_model(self):
        # 读取 LabelEncoder、OneHotEncoder、Tokenizer、embedding_matrix、lstm
        with open(label_save_path, 'rb') as f: self.LABEL_ENCODER = pickle.load(f)
        logger.info(load_log_pattern.format(label_save_path))

        with open(one_hot_save_path, 'rb') as f: self.ONE_HOT_ENCODER = pickle.load(f)
        logger.info(load_log_pattern.format(one_hot_save_path))

        with open(tokenizer_save_path, 'rb') as f: self.TOKENIZER = pickle.load(f)
        logger.info(load_log_pattern.format(tokenizer_save_path))

        with open(embedding_matrix_path, 'rb') as f: self.EMBEDDING_MATRIX = pickle.load(f)
        logger.info(load_log_pattern.format(embedding_matrix_path))

        self.RNN_MODEL = models.load_model(lstm_save_path)
        logger.info(load_log_pattern.format(lstm_save_path))

    def save_model(self):
        # 保存 LabelEncoder、OneHotEncoder、Tokenizer、embedding_matrix
        with open(label_save_path, 'wb') as f: pickle.dump(self.LABEL_ENCODER, f)
        logger.info(save_log_pattern.format(label_save_path))

        with open(one_hot_save_path, 'wb') as f: pickle.dump(self.ONE_HOT_ENCODER, f)
        logger.info(save_log_pattern.format(one_hot_save_path))

        with open(tokenizer_save_path, 'wb') as f:
            pickle.dump(self.TOKENIZER, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(save_log_pattern.format(tokenizer_save_path))

        with open(embedding_matrix_path, 'wb') as f: pickle.dump(self.EMBEDDING_MATRIX, f)
        logger.info(save_log_pattern.format(embedding_matrix_path))

        self.RNN_MODEL.save(lstm_save_path)
        logger.info(save_log_pattern.format(lstm_save_path))