# /usr/bin/env python3
# -*- coding:utf-8 -*-
import sys
import os
import re
import logging
import requests
from lxml import etree
from utility.writer import CSVWriter

sys.path.insert(0, os.path.abspath(os.getcwd()+"/../"))
sys.path.insert(0, os.path.abspath(os.getcwd()+"/../../"))

BASE_URL = 'http://www.teachercn.com'
# BASE_DIR = "/Users/zhengchubin/Desktop/text_classification/"
BASE_DIR = "/home/ai/text_classification/"

CORPUS_DIR = os.path.join(BASE_DIR, 'corpus')
SEG_DIR = os.path.join(BASE_DIR, 'seg')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
GRADE = '年级'
SUBJECT = '科目'
TITLE = '标题'
CONTENT = '内容'
DOC_URL = 'URL'

# 得到文件名
logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)

# 正则-去掉特殊字符
COMPILER = re.compile('[\u3000\xa0]')

def getDocsFromHtml(grade, subject, url):
    ''' 解析年级 http://www.teachercn.com/Jxal/Cysxja/ '''
    res = requests.get(url=url, timeout=60)
    if res.status_code != 200 : return []
    html = etree.HTML(res.content.decode(encoding='gb2312', errors='ignore'))
    rows = html.xpath(r'//td[@valign="top"]')[1]
    rows = rows.xpath(r'./table/tr')
    docs = []
    for row in rows:
        a = row.xpath(r'./td/a')
        if len(a) == 0:
            continue
        a = a[0]
        doc_title = a.xpath(r'text()')[0]
        doc_url = BASE_URL + a.xpath(r'@href')[0]
        doc_content = getContentFromHtml(doc_url)
        docs.append(
            {
                GRADE : grade,
                SUBJECT : subject,
                TITLE : doc_title,
                DOC_URL : doc_url,
                CONTENT : doc_content
             }
        )
        print(doc_title, doc_url)
    return docs

def getContentFromHtml(url, index=5):
    ''' 解析文章 '''
    base = '.'.join(url.split('.')[:-1:1])
    rows = []
    for i in range(index):
        try:
            res = requests.get(url=url, timeout=60)
            res.encoding = 'gb2312'
            if res.status_code != 200:
                continue
            html = etree.HTML(res.content)
            # 获取 p 标签或 a 标签的内容
            rows.extend(html.xpath(r'//p/text() | //p/a/text() | //H3/a/text() | //div/text()' ))
            # print(rows)
            url = base + '_{}.html'.format(i+2)
        except BaseException as e:
            print(e)
    return '  '.join(rows)

def save2csv(fpath, key_values):
    writer = CSVWriter.CSVWriter(fpath)
    Headers = ['年级','科目', '标题', 'URL', '内容']
    writer.write(Headers, key_values)

if __name__ == '__main__':
    getContentFromHtml('http://www.teachercn.com/Jylw/Zzlw/2007/11-4/2007110411412642039.html')