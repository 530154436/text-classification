# /usr/bin/env python3
# -*- coding:utf-8 -*-
from keras.layers import Embedding
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class RNN(object):
    def __init__(self, class_num=9):
        self.class_num = class_num

    def get_keras_embedding(self, max_sequence_length, word2vec, train_embeddings=False):
        """
        加载欲训练的词向量模型作为 Keras 的 Embedding Layer
        """
        weights = word2vec.syn0
        # set `trainable` as `False` to use the pretrained word embedding
        # No extra mem usage here as `Embedding` layer doesn't create any new matrix for weights
        # maweights.shape[0] -> max_features
        # maweights.shape[1] -> embedding_dim
        layer = Embedding(
            input_dim=weights.shape[0], output_dim=weights.shape[1],
            weights=[weights], trainable=train_embeddings, input_length=max_sequence_length
        )
        return layer

    def plot_history(self, history):
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        x = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x, acc, 'b', label='Training acc')
        plt.plot(x, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(x, loss, 'b', label='Training loss')
        plt.plot(x, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()