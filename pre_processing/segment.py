# /usr/bin/env python3
# -*- coding:utf-8 -*-
import jieba
import os
import sys
from data_collecting import common
from utility.reader.CSVReader import CSVReader
from utility.writer.CSVWriter import CSVWriter
sys.path.insert(0, os.path.abspath(os.getcwd()+"/../"))
sys.path.insert(0, os.path.abspath(os.getcwd()+"/../../"))

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

# STOPWORDS = load_stop_words('../dic/stopwords.txt')
STOPWORDS = load_stop_words('dic/stopwords.txt')

def load_from_csv(fpath):
    '''读文件'''
    reader = CSVReader(fpath)
    return reader.read2JsonList()

def save2csv(fpath, Headers, key_values):
    writer = CSVWriter(fpath)
    writer.write(Headers, key_values)

def seg_sentence(sentence, stopwords=STOPWORDS):
    '''对文章进行分词'''
    sentence_seged = jieba.cut(sentence.strip())
    result = []
    for word in sentence_seged:
        # 过滤停用词
        if len(word.strip())<=1:
            continue
        if stopwords is not None and word in stopwords:
            continue
        result.append(word)
    return result

def seg_corpus(corpus_path, user_dic_path, seg_path):
    # 加载词典
    jieba.load_userdict(user_dic_path)
    segs = {}
    types = os.listdir(corpus_path)
    for cate in types:
        dir_name = os.path.join(corpus_path, cate)
        if os.path.isfile(dir_name): continue
        files = os.listdir(dir_name)

        for file in files:
            if file=='.DS_Store': continue
            file_path = os.path.join(dir_name, file)
            documents = load_from_csv(file_path)

            print(file_path)
            for document in documents:
                content = document[CONTENT]
                subject = document[SUBJECT]
                title = document[TITLE]

                # 部分content为空，调用java-TutorialContentExtractor
                seg_content = seg_sentence(content)
                seg_tilte = seg_sentence(title)

                if not seg_content or len(seg_content) < 10: continue

                if subject not in segs:
                    segs[subject] = []
                segs[subject].append({TITLE:' '.join(seg_tilte), CONTENT:' '.join(seg_content), SUBJECT:subject})
    Headers = [SUBJECT, TITLE, CONTENT]
    for subject in segs.keys():
        path = os.path.join(seg_path, '{}.csv'.format(subject))
        save2csv(path, Headers, segs[subject])

if __name__ == '__main__':
    user_dic_path = '../dic/userdic.txt'
    corpus_path = common.CORPUS_DIR
    seg_path = common.SEG_DIR
    # corpus_path = 'D:\文本分类\语料库\数据库'
    seg_corpus(corpus_path, user_dic_path, seg_path)