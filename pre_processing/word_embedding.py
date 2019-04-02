# /usr/bin/env python3
# -*- coding:utf-8 -*-
from pre_processing import segment
from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing
import logging
import zhconv
import jieba
import os.path

# 得到文件名
logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)


def processing_wiki_cn(input, output):
    '''
    处理维基百科中文语料
    :param input:
    :param output:
    :return:
    '''
    logger.info("running %s" % input)

    # 加载词典
    jieba.load_userdict('../dic/userdic.txt')
    stopwords = segment.load_stop_words('../dic/stopwords.txt')

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
    model = Word2Vec(LineSentence(seg_file), workers=multiprocessing.cpu_count(),
                     iter=iter, sg=sg, size=size, window=window, min_count=min_count)

    # 保存模型
    model.save("{}.model".format(output))
    model.wv.save_word2vec_format("{}.vector".format(output), binary=False)
    model.wv.save_word2vec_format("{}.vector.bin".format(output), binary=True)

def load_word2vec(model_path):
    '''
    加载词向量
    '''
    model = Word2Vec.load(model_path)
    # word = model.most_similar("足球")
    # for t in word:
    #     print(t[0], t[1])
    return model


if __name__ == '__main__':
    # BaseDir = '/home/habout/Desktop/text_classification'
    BaseDir = '/Users/zhengchubin/Desktop/text_classification'
    # input = os.path.join(BaseDir, 'zhwiki-latest-pages-articles.xml.bz2')
    # output = os.path.join(BaseDir, 'wiki-zh.seg.txt')
    # processing_wiki_cn(input, output)

    seg = os.path.join(BaseDir, "seg", "seg.txt")
    model = os.path.join(BaseDir, "model", "wiki.zh")

    # 训练
    # train_word2vec(seg, model)

    model = os.path.join(BaseDir, "model", "wiki.zh.model")
    load_word2vec(model)


