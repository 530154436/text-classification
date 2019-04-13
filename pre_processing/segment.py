# /usr/bin/env python3
# -*- coding:utf-8 -*-
import jieba
import os
import zhconv
import pandas as pd
from gensim.corpora import WikiCorpus
from utility.reader.CSVReader import CSVReader
from utility.writer.CSVWriter import CSVWriter
from utility.log_util import  logger
from utility import pandas_util
from config import logger,CORPUS_DIR,SEG_DIR
import config

CHINES_CHARSETS = ["\u200b","\u2000","\u206F","\u2E00","\u2E7F","\u3000",'\u3000'
                   "\u303F","\uff01","\uff02","\uff08","\uff09","\uff0c",'\xa0'
                   "\uff0e","\uff0f","\uff1a","\uff1b","\uff1f","\u0020","\u00BF"]
GRADE = '年级'
SUBJECT = '科目'
TITLE = '标题'
CONTENT = '内容'
DOC_URL = 'URL'

def load_stop_words(filePath):
    '''加载停用词词典和中文特殊字符'''
    stopwords = set([line.strip() for line in open(filePath, 'r', encoding='utf-8').readlines()])
    stopwords.union(set(CHINES_CHARSETS))
    return stopwords

# 加载词典
STOPWORDS = load_stop_words(os.path.join(CORPUS_DIR, 'dic/stopwords.txt'))
jieba.load_userdict(os.path.join(CORPUS_DIR, 'dic/userdic.txt'))

def load_from_csv(fpath):
    '''读文件'''
    reader = CSVReader(fpath)
    return reader.read2JsonList()

def save2csv(fpath, Headers, key_values):
    writer = CSVWriter(fpath)
    writer.write(Headers, key_values)

def seg_sentence(sentence):
    '''对文章进行分词'''
    sentence_seged = jieba.cut(sentence.strip())
    result = []
    for word in sentence_seged:
        # 过滤停用词
        if len(word.strip())<=1:
            continue
        if STOPWORDS is not None and word in STOPWORDS:
            continue
        result.append(word)
    return result

def seg_wiki_cn():
    '''
    处理维基百科中文语料
    '''
    input = os.path.join(CORPUS_DIR, 'wiki.zh.xml.bz2')
    output = os.path.join(SEG_DIR, 'wiki.zh.seg.txt')
    logger.info("running %s" % input)

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
            seged_sentence = seg_sentence(convert_sentence)

            out.write(space.join(seged_sentence)+"\n")
            i = i+1
            if (i % 1000 == 0):
                logger.info("Saved "+str(i)+" articles.")
        logger.info("Finished Saved "+str(i)+" articles.")

def seg_corpus():
    segs = {}
    types = os.listdir(CORPUS_DIR)
    count = 0
    for cate in types:
        if cate not in ['ja', 'lw', 'extend']:
            continue
        dir_name = os.path.join(CORPUS_DIR, cate)
        if os.path.isfile(dir_name): continue
        files = os.listdir(dir_name)

        for file in files:
            if file=='.DS_Store': continue
            file_path = os.path.join(dir_name, file)
            documents = load_from_csv(file_path)

            for document in documents:
                count += 1
                content = document[CONTENT]
                subject = document[SUBJECT]
                title = document[TITLE]

                # 部分content为空，调用java-TutorialContentExtractor
                seg_content = seg_sentence(content)
                seg_tilte = seg_sentence(title)

                if not seg_content or len(seg_content) < 10: continue

                if count % 100==0:
                    logger.info("分词中 {}: {}".format(file_path, count))

                if subject not in segs:
                    segs[subject] = []
                segs[subject].append({TITLE:' '.join(seg_tilte), CONTENT:' '.join(seg_content), SUBJECT:subject})
    Headers = [SUBJECT, TITLE, CONTENT]
    for subject in segs.keys():
        path = os.path.join(SEG_DIR, '{}.csv'.format(subject))
        save2csv(path, Headers, segs[subject])

def save2txt():
    segs_path = [os.path.join(SEG_DIR, "{}.csv".format(i)) for i in config.SUBJECTS]
    with open(os.path.join(SEG_DIR,'seg.txt'), 'w') as f:
        df = []
        for seg in segs_path:
            print(seg)
            seg_df = pandas_util.readCsv2PdByChunkSize(seg, [CONTENT])
            df.append(seg_df)
        pd_concat = pd.concat(df)
        for i in pd_concat[CONTENT]:
            f.write(i)
            f.write('\n')

if __name__ == '__main__':
    # seg_corpus()
    save2txt()
    # seg_wiki_cn()
