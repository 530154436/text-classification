# /usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import argparse
from pre_processing import word_embedding
from config import SEG_DIR
import config

def main(seg_file=None):
    if seg_file is None:
        seg_file = os.path.join(SEG_DIR, "merge.txt")
    word_embedding.train_word2vec(
        seg_file,
        number_cores=config.NUM_CORES,
        sg=config.SG,
        size=config.SIZE,
        window=config.WINDOW,
        min_count=config.MIN_COUNT,
        iter=config.ITER
    )

def parse():
    '''
    解析命令行参数
    '''
    parser = argparse.ArgumentParser()
    # word2vec
    parser.add_argument("--sg", dest='SG',   required=True, type=int)
    parser.add_argument("--size", dest='SIZE', required=True, type=int)
    parser.add_argument("--iter", dest='ITER', required=True, type=int )
    parser.add_argument("--file_path", dest='FILE_PATH', required=False, type=int)
    parser.add_argument("--num_cores", dest='NUM_CORES', required=False, type=int)
    parser.add_argument("--window", dest='WINDOW', required=False, type=int)
    parser.add_argument("--min_count", dest='MIN_COUNT', required=False, type=int)

    args = parser.parse_args()
    config.SG = args.SG
    config.SIZE = args.SIZE
    config.ITER = args.ITER
    if args.NUM_CORES:
        config.NUM_CORES = args.NUM_CORES
    if args.WINDOW:
        config.WINDOW = args.WINDOW
    if args.MIN_COUNT:
        config.MIN_COUNT = args.MIN_COUNT
    return args.FILE_PATH

if __name__ == '__main__':
    seg_file = parse()
    main(seg_file)
