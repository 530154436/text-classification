#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import config
from app import create_app
from model.LSTM import TextRNN_LSTM
from pre_processing import segment

text_rnn = TextRNN_LSTM(class_num=config.CLASS_NUM)
text_rnn.load_model()
segmenter = segment

# https://github.com/xiiiblue/flask-adminlte-scaffold
# https://v3.bootcss.com/css/?#forms
app,files = create_app(os.getenv('FLASK_CONFIG') or 'default')

if __name__ == '__main__':
    app.run(debug=False)
