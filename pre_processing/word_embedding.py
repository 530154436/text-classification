# /usr/bin/env python3
# -*- coding:utf-8 -*-
from gensim.corpora import WikiCorpus
import logging
import os.path

def processing_wiki_cn(input, output):
    '''
    处理维基百科中文语料
    :param input:
    :param output:
    :return:
    '''
    # 得到文件名
    logger = logging.getLogger(os.path.basename(input))
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(input))

    space = " "
    i = 0
    with open(output, mode='w', encoding='utf-8') as out:
        # gensim里的维基百科处理类WikiCorpus
        wiki =WikiCorpus(input, lemmatize=False, dictionary=[])
        # 通过get_texts将维基里的每篇文章转换位1行text文本，并且去掉了标点符号等内容
        for text in wiki.get_texts():
            out.write(space.join(text) + "\n")
            i = i+1
            if (i % 10000 == 0):
                logger.info("Saved "+str(i)+" articles.")
        logger.info("Finished Saved "+str(i)+" articles.")

if __name__ == '__main__':
    BaseDir = '~/D/text_classification'
    input = os.path.join(BaseDir, 'zhwiki-latest-pages-articles.xml')
    output = os.path.join(BaseDir, 'wiki-zh.txt')
    processing_wiki_cn(input, output)