# /usr/bin/env python3
# -*- coding:utf-8 -*-
import sys
import os
from data_collecting import common

sys.path.insert(0, os.path.abspath(os.getcwd()+"/../"))
sys.path.insert(0, os.path.abspath(os.getcwd()+"/../../"))

'''
中教网-教案 http://www.teachercn.com/Jxal/
'''

SUBJECT_URL = {
    '语文' : 'yw',
    '数学' : 'sx',
    '英语' : 'yy',
    '政治' : 'zz',
    '物理' : 'wl',
    '化学' : 'hx',
    '地理' : 'dl',
    '历史' : 'ls',
    '生物' : 'sw'
}

GRADE_URL = {
    '初中一年级' :'Cy',
    '初中二年级': 'Ce',
    '初中三年级': 'Cs',
    '高中一年级': 'Gy',
    '高中二年级': 'Ge',
    '高中三年级': 'Gs',
}

JIAO_AN = 'ja'
SAVE_DIR = os.path.join(common.CORPUS_DIR, JIAO_AN)

# 全局统计量
COUNT = 0

def getURL( grade, subject, type='Jxal', index = 1):
    ''' 构造URL '''
    baseURL = '{}/{}/{}{}ja/index{}.html'
    if index <= 1:
        baseURL = baseURL.format(common.BASE_URL, type, grade, subject, '')
    else:
        baseURL = baseURL.format(common.BASE_URL, type, grade, subject, '_' + str(index))
    return baseURL

def main(index = 5):
    # index = 'http://www.teachercn.com/zxyw/Html/CZYNJYWJA/index{}.html'
    count = 0
    for ks, vs in SUBJECT_URL.items():
        for kg,vg in GRADE_URL.items():
            print(kg, ks)
            documents = []
            for i in range(index):
                url = getURL(vg, vs, index=i+1)
                print(url)
                documents.extend(common.getDocsFromHtml(kg, ks, url))
            filePath = os.path.join(SAVE_DIR, '{}_{}_{}.csv'.format(ks, kg, JIAO_AN))
            common.save2csv(filePath, documents)
            count += len(documents)
            print('\n共爬取 {} 篇教案\n'.format(count))

if __name__ == '__main__':
    main()