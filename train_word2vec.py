# /usr/bin/env python3
# -*- coding:utf-8 -*-
import os
from pre_processing import word_embedding
from data_collecting.common import SEG_DIR

seg_file = os.path.join(SEG_DIR, "merge.txt")
word_embedding.train_word2vec(seg_file, cpu_count=8, sg=0, size=50, window=5, min_count=3, iter=50)
word_embedding.train_word2vec(seg_file, cpu_count=8, sg=0, size=150, window=5, min_count=3,iter=50)