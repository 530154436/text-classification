# /usr/bin/env python3
# -*- coding:utf-8 -*-

from keras import Sequential
from keras.layers import LSTM,Dense,Embedding,Dropout
from keras.optimizers import Adam
from model.RNN import RNN
from keras.utils import plot_model

class TextRNN_LSTM(RNN):
    def __init__(self, embedding_matrix, class_num=9):
        super(TextRNN_LSTM, self).__init__(embedding_matrix, class_num)

    def get_model(self, max_sequence_length,
                  lstm_drop=0.2,
                  lstm_num=200,
                  dense_num=200,
                  last_activation='softmax',
                  loss='categorical_crossentropy',
                  metrics = ('accuracy', 'categorical_accuracy')
                  ):
        # embedding_layer = self.get_keras_embedding()
        max_features,embedding_dim  = self.embedding_matrix.shape
        embedding_layer = Embedding(max_features,  # embedding layer
                                    embedding_dim,
                                    weights=[self.embedding_matrix],
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