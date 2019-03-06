#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import csv
import sys
csv.field_size_limit(100000000)

class CSVReader:
    def __init__(self, fpath):
        self.fpath = fpath
        self.file = None

    def openFile(self, encoding='utf-8'):
        try:
            self.file = open(self.fpath, 'r', encoding=encoding)
        except Exception as e:
            print('Error while opening file.', str(e))

    def closeFile(self):
        try:
            self.file.close()
        except Exception as e:
            print('Error while closing file.', str(e))

    def read(self):
        '''

        :param headers:
        :param datas:
        :return:
        '''
        if self.file is None:
            self.openFile()
        f_csv = csv.reader(self.file)
        return f_csv

    def read2JsonList(self, fieldTypes=None):
        '''
        csv转json
        :param fieldTypes: 标题列对应的数据类型 {标题:类型}，默认均为str
        :return:  json列表
        '''
        if self.file is None:
            self.openFile()
        csv_rows = []
        reader = csv.DictReader(self.file)
        title = reader.fieldnames
        count = 0
        for row in reader:
            try:
                result = {}
                for i in range(len(title)):
                    fieldName = title[i]
                    value = row[title[i]]
                    if fieldTypes is not None and fieldName in fieldTypes:  # 强制转换类型
                        value = fieldTypes[fieldName](value)
                    result[fieldName] = value
                csv_rows.append(result)
                count += 1
                if count % 100 == 0:
                    print("读取csv文件: {}".format(count))
            except Exception as e:
                print(e, row)
        return csv_rows



