# /usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import sys
from data_collecting.common import logger,BASE_DIR
from pre_processing import segment
from gensim.corpora import WikiCorpus
from gensim.models import word2vec, KeyedVectors
from gensim.models.word2vec import LineSentence
import multiprocessing
import logging
import zhconv
import jieba
import os.path

sys.path.insert(0, os.path.abspath(os.getcwd()+"/../"))
sys.path.insert(0, os.path.abspath(os.getcwd()+"/../../"))

def processing_wiki_cn(input, output):
    '''
    处理维基百科中文语料
    :param input:
    :param output:
    :return:
    '''
    logger.info("running %s" % input)

    # 加载词典
    jieba.load_userdict('dic/userdic.txt')
    stopwords = segment.load_stop_words('dic/stopwords.txt')

    space = " "
    i = 0
    with open(output, mode='w', encoding='utf-8') as out:
        # gensim里的维基百科处理类WikiCorpus
        wiki =WikiCorpus(input, lemmatize=False, dictionary=[])
        # 通过get_texts将维基里的每篇文章转换位1行text文本，并且去掉了标点符号等内容
        for text in wiki.get_texts():
            sentence = space.join(text)
            # 繁简转换
            convert_sentence = zhconv.convert(sentence, 'zh-hans')
            # 分词
            seg_sentence = segment.seg_sentence(convert_sentence, stopwords=stopwords)

            out.write(space.join(seg_sentence)+"\n")
            i = i+1
            if (i % 10000 == 0):
                logger.info("Saved "+str(i)+" articles.")
        logger.info("Finished Saved "+str(i)+" articles.")

def train_word2vec(seg_file, output, sg=1, size=300, window=5, min_count=3,iter=100):
    '''
    训练词向量模型
    sg  default (`sg=0`), CBOW
        Otherwise (`sg=1`), skip-gram
    '''
    logger.info("running %s" % seg_file)

    # 训练 skip-gram 模型
    model = word2vec.Word2Vec(LineSentence(seg_file), workers=multiprocessing.cpu_count(),
                     iter=iter, sg=sg, size=size, window=window, min_count=min_count)

    # 保存模型
    # model.save("{}.sg{}.size{}.iter{}".format(output, sg, size, iter))
    model.wv.save_word2vec_format("{}.sg{}.size{}.iter{}".format(output, sg, size, iter), binary=False)
    model.wv.save_word2vec_format("{}.sg{}.size{}.iter{}.bin".format(output, sg, size, iter), binary=True)

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
    model_path = os.path.join(BASE_DIR, "model", "wiki.zh.vector")
    model = load_word2vec(model_path)
    word = "长沙"
    print(model['长沙'])
    print(word in model)
    print('lala' in model)

if __name__ == '__main__':
    # BaseDir = '/home/habout/Desktop/text_classification'
    BaseDir = '/Users/zhengchubin/Desktop/text_classification'
    # input = os.path.join(BaseDir, 'zhwiki-latest-pages-articles.xml.bz2')
    # output = os.path.join(BaseDir, 'wiki-zh.seg.txt')
    # processing_wiki_cn(input, output)

    seg = os.path.join(BaseDir, "seg", "seg.txt")
    model = os.path.join(BaseDir, "vector")

    # 训练
    # train_word2vec(seg, model, sg=1, size=300, window=5, min_count=3,iter=100)

    test()

