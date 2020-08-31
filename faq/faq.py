#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-08-26 16:22:08
# @Author  : Kaiyan Zhang (minekaiyan@gmail.com)
# @Link    : https://github.com/iseesaw
# @Version : 1.0.0

import time
import logging
import numpy as np

from utils import load_json, cos_dist
from enc_client import EncodeClient

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO)
logger = logging.getLogger(__name__)


def init():
    '''加载预处理数据
    '''
    faq_data = load_json('hflqa/faq.json')
    corpus_mat = np.load('hflqa/corpus_mat.npy')
    topics = load_json('hflqa/topics.json')
    corpus_mat_norm = np.linalg.norm(corpus_mat)
    return faq_data, topics, corpus_mat, corpus_mat_norm


print('start loading')
faq_data, topics, corpus_mat, corpus_mat_norm = init()
client = EncodeClient()
print('end loading...')


def query():
    '''测试
    '''
    while True:
        enc = client.encode([input('Enter: ')])
        t1 = time.time()
        scores = cos_dist(np.squeeze(enc, axis=0), corpus_mat, corpus_mat_norm)
        max_index = np.argmax(scores)

        topic = topics[max_index]

        resp = faq_data[topic]['resp']
        print(np.random.choice(resp, 1)[0], time.time() - t1)


if __name__ == '__main__':
    query()
