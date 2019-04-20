# /usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import logging
from scrapy import signals
from scrapy.contrib.exporter import CsvItemExporter
from jiaoan.spiders.JiaoAnSpider import CONTENT,TITLE,DOC_URL,GRADE,subjects

BASE_DIR = "/Users/zhengchubin/Desktop/text_classification/corpus/extend"

class JiaoAnPipeline(object):
    @classmethod
    def from_crawler(cls, crawler):
        pipeline = cls()
        crawler.signals.connect(pipeline.spider_opened, signals.spider_opened)
        crawler.signals.connect(pipeline.spider_closed, signals.spider_closed)
        return pipeline

    def spider_opened(self, spider):
        self.senior = open(os.path.join(BASE_DIR, '高中.csv'), 'wb')
        self.senior_exporter = CsvItemExporter(self.senior)
        self.senior_exporter.start_exporting()

        self.junior = open(os.path.join(BASE_DIR, '初中.csv'), 'wb')
        self.junior_exporter = CsvItemExporter(self.junior)
        self.junior_exporter.start_exporting()

    def spider_closed(self, spider):
        self.senior_exporter.finish_exporting()
        self.senior.close()

        self.junior_exporter.finish_exporting()
        self.junior.close()

    def process_item(self, item, spider):
        '''
        数据清洗
        '''
        if TITLE in item:
            item[TITLE] = item[TITLE]
            logging.info('标题:{}'.format(item[TITLE]))
        if CONTENT in item:
            content = item[CONTENT].replace('\n','')
            item[CONTENT] = content
            logging.info('内容:{} ...'.format(item[CONTENT][:70]))
        # if '高中' in item[GRADE]:
        #     self.senior_exporter.export_item(item)
        # if '初中' in item[GRADE]:
        #     self.junior_exporter.export_item(item)
        return item

