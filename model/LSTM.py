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
from config import MODEL_DIR,logger, SG, SIZE, ITER, LSTM_NUM, MAX_SEQUENCE_LEN,BATCH_SIZE, DENSE_NUM, \
                LSTM_DROP, LSTM,LABEL_ENCODER,ONE_HOT_ENCODER, TOKENIZER, RNN_MODEL

save_log_pattern = '{} 已保存.'
load_log_pattern = '{} 已加载.'

label_save_path = os.path.join(MODEL_DIR, 'label_encoder.pickle')
one_hot_save_path = os.path.join(MODEL_DIR, 'one_hot_encoder.pickle')
tokenizer_save_path = os.path.join(MODEL_DIR, 'text_tokenizer.pickle')
embedding_matrix_path = os.path.join(MODEL_DIR, 'embedding_matrix.pickle')
lstm_save_path = os.path.join(MODEL_DIR,'vsg{}.vs{}.vi{}.ln{}.dn{}.ld{}.{}.h5'.format(
    SG, SIZE, ITER, LSTM_NUM, DENSE_NUM, LSTM_DROP, LSTM))

class TextRNN_LSTM(RNN):
    def __init__(self, class_num=9):
        super(TextRNN_LSTM, self).__init__(class_num)

    def get_model(self,
                  embedding_matrix,
                  max_sequence_length,
                  lstm_drop=0.2,
                  lstm_num=200,
                  dense_num=200,
                  last_activation='softmax',
                  loss='categorical_crossentropy',
                  metrics = ('accuracy', 'categorical_accuracy')
                  ):
        # embedding_layer = self.get_keras_embedding()
        max_features,embedding_dim  = embedding_matrix.shape
        embedding_layer = Embedding(max_features,  # embedding layer
                                    embedding_dim,
                                    weights=[embedding_matrix],
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
        # plot_model(model, to_file='pic/lstm.png')
        return model

    def metric(self):
        '''
        评估
        '''
        # 加载数据
        dfs = loadData([os.path.join(config.SEG_DIR, "{}.csv".format(i)) for i in config.SUBJECTS])
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
        x = TOKENIZER.texts_to_sequences(x)
        x = pad_sequences(x, padding='post', maxlen=MAX_SEQUENCE_LEN)
        y = RNN_MODEL.predict(x, batch_size=BATCH_SIZE)
        for row in y:
            new_y.append((row == row.max(axis=0)) + 0)
        y = ONE_HOT_ENCODER.inverse_transform(new_y)
        y = LABEL_ENCODER.inverse_transform(y.ravel())
        return y

    def load_model(self):
        # 读取 LabelEncoder、OneHotEncoder、Tokenizer、embedding_matrix、lstm
        with open(label_save_path, 'rb') as f: config.LABEL_ENCODER = pickle.load(f)
        logger.info(load_log_pattern.format(label_save_path))

        with open(one_hot_save_path, 'rb') as f: config.ONE_HOT_ENCODER = pickle.load(f)
        logger.info(load_log_pattern.format(one_hot_save_path))

        with open(tokenizer_save_path, 'rb') as f: config.TOKENIZER = pickle.load(f)
        logger.info(load_log_pattern.format(tokenizer_save_path))

        with open(embedding_matrix_path, 'rb') as f: config.EMBEDDING_MATRIX = pickle.load(f)
        logger.info(load_log_pattern.format(embedding_matrix_path))

        config.RNN_MODEL = models.load_model(lstm_save_path)
        logger.info(load_log_pattern.format(lstm_save_path))

    def save_model(self):
        # 保存 LabelEncoder、OneHotEncoder、Tokenizer、embedding_matrix
        with open(label_save_path, 'wb') as f: pickle.dump(config.LABEL_ENCODER, f)
        logger.info(save_log_pattern.format(label_save_path))

        with open(one_hot_save_path, 'wb') as f: pickle.dump(config.ONE_HOT_ENCODER, f)
        logger.info(save_log_pattern.format(one_hot_save_path))

        with open(tokenizer_save_path, 'wb') as f:
            pickle.dump(config.TOKENIZER, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(save_log_pattern.format(tokenizer_save_path))

        with open(embedding_matrix_path, 'wb') as f: pickle.dump(config.EMBEDDING_MATRIX, f)
        logger.info(save_log_pattern.format(embedding_matrix_path))

        config.RNN_MODEL.save(lstm_save_path)
        logger.info(save_log_pattern.format(lstm_save_path))