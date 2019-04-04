# /usr/bin/env python3
# -*- coding:utf-8 -*-
from keras.datasets import imdb
from keras import Sequential
from keras import preprocessing
from keras.layers import Flatten,Dense,Embedding

# Embedding Layer
# 输入 (samples, sequence_length)
# 返回 (samples, sequence_length, embedding_dimensionality)

# 加载 IMDB 数据，准备用于 Embedding 层
max_features = 10000  # 作为特征的单词个数
maxlen = 20
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(x_train)
print(y_train)
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
print(x_train)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

# 在 IMDB 数据上使用 Embedding 层和分类器
model = Sequential()
model.add(Embedding(10000, 8, input_length=maxlen))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
