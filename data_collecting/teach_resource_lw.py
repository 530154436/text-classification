# /usr/bin/env python3
# -*- coding:utf-8 -*-
import sys
import os
from data_collecting import common

sys.path.insert(0, os.path.abspath(os.getcwd()+"/../"))
sys.path.insert(0, os.path.abspath(os.getcwd()+"/../../"))

SUBJECT_URL = {
    '语文' : 'yw',
    '数学' : 'sx',
    '英语' : 'yy',
    '物理' : 'wl',
    '化学': 'hx',

    '地理': 'dl',
    '历史': 'ls',
    '政治' : 'zz',
    '生物' : 'sw'
}

GRADE_URL = {
    '初中' :'Cz',
    '高中': 'Gz'
}

LUN_WEN = 'lw'
SAVE_DIR = os.path.join(common.BASE_DIR, LUN_WEN)

# 全局统计量
COUNT = 0

def getURL( grade, subject, type='Jylw', index = 1):
    ''' 构造URL '''
    baseURL = '{}/{}/{}/index{}.html'

    sub = '{}{}lw'.format(grade, subject)
    if subject in ['dl', 'ls', 'zz', 'sw']:
        sub = '{}lw'.format(subject.capitalize())
    if index <= 1:
        baseURL = baseURL.format(common.BASE_URL, type, sub, '')
    else:
        baseURL = baseURL.format(common.BASE_URL, type, sub, '_' + str(index))
    return baseURL

def main(index = 10):
    count = 0
    for k_subject, v_subject in SUBJECT_URL.items():
        for k_grade,v_grade in GRADE_URL.items():
            if v_subject in ['dl', 'ls', 'zz', 'sw']:
                k_grade = '中学'
                v_grade = ''
            print(k_grade, k_subject)
            documents = []
            for i in range(index):
                url = getURL(v_grade, v_subject, index=i+1)
                print(url)
                documents.extend(common.getDocsFromHtml(k_grade, k_subject, url))
            filePath = os.path.join(SAVE_DIR, '{}_{}_{}.csv'.format(k_subject, k_grade, LUN_WEN))
            common.save2csv(filePath, documents)
            count += len(documents)
            print('\n共爬取 {} 篇教案\n'.format(count))
            if v_subject in ['dl', 'ls', 'zz', 'sw']:
                break

if __name__ == '__main__':
    main()