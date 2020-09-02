#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-08-26 16:22:08
# @Author  : Kaiyan Zhang (minekaiyan@gmail.com)
# @Link    : https://github.com/iseesaw
# @Version : 1.0.0
import time
import logging
import numpy as np

from utils import load_json, cos_sim, get_model

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO)
logger = logging.getLogger(__name__)


def init_data():
    """加载数据
    """
    faq_data = load_json('hflqa/faq.json')
    corpus_mat = np.load('hflqa/corpus_mat.npy')
    topics = load_json('hflqa/topics.json')
    corpus_mat_norm = np.linalg.norm(corpus_mat)
    return faq_data, topics, corpus_mat, corpus_mat_norm


print('start loading')
faq_data, topics, corpus_mat, corpus_mat_norm = init_data()
model = get_model('./output/transformers-merge3-bert-6L')
print('end loading...')


def query():
    """输入测试
    """
    while True:
        enc = model.encode(input('Enter: '))
        t1 = time.time()
        scores = cos_sim(enc, corpus_mat, corpus_mat_norm)
        max_index = np.argmax(scores)

        topic = topics[max_index]

        resp = faq_data[topic]['resp']
        print(np.random.choice(resp, 1)[0], time.time() - t1)


if __name__ == '__main__':
    query()
