# /usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import re
import numpy as np
BASE_DIR = "/Users/zhengchubin/Desktop/text_classification/"
COMPILER = re.compile('.+acc: (.+) -.+')
ECHO = re.compile('Epoch ([0-9]+)/([0-9]+)')

'''
    日志分析-统计最后一次迭代的准确率
'''

for i in range(20):
    file = os.path.join(BASE_DIR, 'logs/lstm{}.log'.format(i))
    if not os.path.exists(file):
        continue
    log = ''
    with open(file, mode='r') as f:
        last = False
        for line in f:
            echos=ECHO.findall(line)
            if len(echos)>0 and (int(echos[0][1])-int(echos[0][0]))==0 :
                last = True
            if not last:
                continue
            log += line
        new_str = [ float(i) for i in COMPILER.findall(log)]

    if new_str:
        print(file, np.average(new_str))