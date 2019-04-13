# /usr/bin/env python3
# -*- coding:utf-8 -*-
import logging

# 日志
logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)