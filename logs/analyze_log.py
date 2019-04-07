# /usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import re
import numpy as np
from data_collecting.common import BASE_DIR
COMPILER = re.compile('.+acc: (.+) -.+')

'''
    日志分析
'''

for i in range(20):
    file = os.path.join(BASE_DIR, 'logs/lstm{}.log'.format(i))
    if not os.path.exists(file):
        continue
    log = ''
    with open(file, mode='r') as f:
        last = False
        for line in f:
            if '200/200' in line :
                last = True
            if not last:
                continue
            log += line
        new_str = [ float(i) for i in COMPILER.findall(log)]

    if new_str:
        print(file, np.average(new_str))