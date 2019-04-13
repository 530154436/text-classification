# /usr/bin/env python3
# -*- coding:utf-8 -*-
import scrapy

class MyItem(scrapy.Item):
    科目 = scrapy.Field()           # 科目
    年级 = scrapy.Field()            # 年级
    标题 = scrapy.Field()         # 标题
    内容 = scrapy.Field()           # 内容
    url = scrapy.Field()               # url