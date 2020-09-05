#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-08-26 16:22:08
# @Author  : Kaiyan Zhang (minekaiyan@gmail.com)
# @Link    : https://github.com/iseesaw
# @Version : 1.0.0
import time
import logging
import numpy as np

from utils import load_json, cos_sim
from transformers_encoder import TransformersEncoder

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO)
logger = logging.getLogger(__name__)


def init_data(model):
    """加载数据
    """
    faq_data = load_json('hflqa/faq.json')
    posts, topics = [], []
    for topic, qas in faq_data.items():
        for post in qas['post']:
            posts.append(post)
            topics.append(topic)

    encs = model.encode(posts, show_progress_bar=True)
    corpus_mat = encs.numpy()
    corpus_mat_norm = np.linalg.norm(corpus_mat)
    return faq_data, topics, corpus_mat, corpus_mat_norm


print('start loading')
model_path = './output/transformers-merge3-bert/'
model = TransformersEncoder(model_path)
faq_data, topics, corpus_mat, corpus_mat_norm = init_data(model)
print('end loading...')


def query():
    """输入测试
    """
    while True:
        enc = model.encode([input('Enter: ')])
        t1 = time.time()
        scores = cos_sim(np.squeeze(enc, axis=0), corpus_mat, corpus_mat_norm)
        max_index = np.argmax(scores)

        topic = topics[max_index]

        resp = faq_data[topic]['resp']
        print(np.random.choice(resp, 1)[0], time.time() - t1)


if __name__ == '__main__':
    query()
