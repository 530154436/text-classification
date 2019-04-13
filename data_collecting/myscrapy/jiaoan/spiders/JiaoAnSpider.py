# /usr/bin/env python3
# -*- coding:utf-8 -*-
import scrapy
import logging
from jiaoan.item import MyItem

GRADE = '年级'
SUBJECT = '科目'
TITLE = '标题'
CONTENT = '内容'
DOC_URL = 'URL'

subject_id = {
    "c129":"政治",
    "c125":"历史",
    "c123":"地理",
    "c120":"生物",
    "c124":"化学",
    "c122":"物理",
    "c121":"英语",
    "c119":"数学",
    "c118":"语文"
}

grade_id = {
    "p104":"高中",
    "p103":"初中"
}

subjects = subject_id.keys()
domain = "http://zsd.jzb.com"

class JiaoAnSpider(scrapy.Spider):
    name = "jiaoan"                 # 爬虫名字，通过反射找到
    allowed_domains = [domain]
    url = domain+ '/ja/{}{}/?pg={}'
    start_urls = []
    # 减慢爬取速度0.5s
    download_delay = 0.3
    for subject in subject_id.keys():
        for grade in grade_id.keys():
            # 初中没有政治
            if subject_id[subject]=='政治' and grade_id[grade]=='初中':
                continue
            for page_index in range(1, 201):
                newURL = url.format(grade, subject, page_index*10)
                start_urls.append(newURL)

    def parse(self, response):
        span = response.xpath('//div[@class="factor_Box"]/span')
        subject = span.xpath('./em/text()').extract()
        grade = span.xpath('./a/@href').extract()

        for href in response.xpath('//div[@class="teach_Lst"]/ul/li/h3/a/@href'):
            url = domain+href.extract()
            item = MyItem()
            item[DOC_URL] = url
            item[SUBJECT] = ''.join(subject)
            if 'p104' in ''.join(grade):
                item[GRADE] = '高中'
            if 'p103' in ''.join(grade):
                item[GRADE] = '初中'
            # dont_filter 使 requests 不被过滤:
            yield  scrapy.Request(url, meta={'item':item}, callback=self.parse_dir_contents, dont_filter=True)

    def parse_dir_contents(self, response):
        item = response.meta['item']
        title = response.xpath('//title/text()').extract()
        content = response.xpath('//div[starts-with(@class, "article_Con")]/pre/text()').extract()
        item[TITLE] = ''.join(title)
        item[CONTENT] = ''.join(content)
        yield item