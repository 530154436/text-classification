# /usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import config
from sklearn import metrics
from config import SEG_DIR, SUBJECTS
from pre_processing import dataset
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

def train_tradiction():
    segs_path = [os.path.join(SEG_DIR, "{}_unique.csv".format(i)) for i in SUBJECTS]
    # segs_path.extend([os.path.join(SEG_DIR, "{}_extend.csv".format(i)) for i in SUBJECTS])
    # 1. 加载数据集
    dfs = dataset.loadData(segs_path, sample_num=300)

    # 2. 划分数据集
    x_train, x_test, y_train, y_test = dataset.splitData(dfs, test_size=0.3)

    # 3.文本转特征向量 tf-idf
    tf_idf =  TfidfVectorizer(analyzer='word').fit(x_train)
    X_train_tfi_df = tf_idf.transform(x_train)
    X_test_tf_idf = tf_idf.transform(x_test)

    # 4. 评测
    # 4.1 朴素贝叶斯
    nb = MultinomialNB()
    nb.fit(X_train_tfi_df, y_train)
    nb_predict = nb.predict(X_test_tf_idf)

    print('----------------------- 朴素贝叶斯 -----------------------')
    print (classification_report(y_test, nb_predict))
    print("平均准确率\n", metrics.accuracy_score(y_test, nb_predict))
    print('----------------------- 朴素贝叶斯 -----------------------')

    # 4.2 逻辑斯蒂回归
    lr = LogisticRegression(multi_class='auto', solver='lbfgs')
    lr.fit(X_train_tfi_df, y_train)
    lr_predict = lr.predict(X_test_tf_idf)

    print('----------------------- 逻辑斯蒂回归 -----------------------')
    print (classification_report(y_test, lr_predict))
    print("平均准确率\n", metrics.accuracy_score(y_test, lr_predict))
    print('----------------------- 逻辑斯蒂回归 -----------------------')

if __name__ == '__main__':
    train_tradiction()