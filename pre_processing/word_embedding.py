# /usr/bin/env python3
# -*- coding:utf-8 -*-
import os
from data_collecting.common import logger,SEG_DIR,MODEL_DIR
from gensim.models import word2vec, KeyedVectors
from gensim.models.word2vec import LineSentence
import multiprocessing
import logging
import os.path

def train_word2vec(seg_file, cpu_count=None, sg=0, size=300, window=5, min_count=3,iter=100):
    '''
    训练词向量模型
    sg  default (`sg=0`), CBOW
        Otherwise (`sg=1`), skip-gram
    '''
    logger.info("running %s" % seg_file)
    if not cpu_count:
        cpu_count = multiprocessing.cpu_count()

    # 训练 skip-gram 模型
    model = word2vec.Word2Vec(LineSentence(seg_file), workers=cpu_count,
                     iter=iter, sg=sg, size=size, window=window, min_count=min_count)
    model.wv.save_word2vec_format(os.path.join(MODEL_DIR, "vector.sg{}.size{}.iter{}.bin".format(sg, size, iter)), binary=True)

def load_word2vec(model_path, binary=False):
    '''
    加载词向量
    '''
    logger.info("加载 {} ...".format(model_path))
    model = KeyedVectors.load_word2vec_format(model_path, binary=binary)
    logger.info("{} 加载完成.".format(model_path))
    return model

def getVec(word, model):
    if word in model:
        return model[word]
    else:
        logging.info("{}不在词表中.".format(word))

def test():
    load_word2vec(os.path.join(MODEL_DIR, "vector.sg0.size300.iter50.bin"))
    model_path = os.path.join(MODEL_DIR, "wiki.zh.vector")
    model = load_word2vec(model_path)
    word = "长沙"
    print(model['长沙'])
    print(word in model)
    print('lala' in model)

if __name__ == '__main__':
    seg_file = os.path.join(SEG_DIR, "merge.txt")
    train_word2vec(seg_file, cpu_count=8, sg=0, size=150, window=5, min_count=3,iter=50)
    train_word2vec(seg_file, cpu_count=8, sg=0, size=50, window=5, min_count=3, iter=50)