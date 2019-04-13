# -*- coding: utf-8 -*-

BOT_NAME = 'jiaoan'
SPIDER_MODULES = ['jiaoan.spiders']
NEWSPIDER_MODULE = 'jiaoan.spiders'
USER_AGENT = 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.162 Safari/537.36'
ROBOTSTXT_OBEY = True
FEED_EXPORT_ENCODING = 'utf-8'

ITEM_PIPELINES = {
    'jiaoan.pipelines.JiaoAnPipeline': 300
}