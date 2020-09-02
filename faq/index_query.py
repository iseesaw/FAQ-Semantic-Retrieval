#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-08-26 01:27:04
# @Author  : Kaiyan Zhang (minekaiyan@gmail.com)
# @Link    : https://github.com/iseesaw
# @Version : 1.0.0
import numpy as np

from utils import load_json, save_json, get_model


def index_query():
    """对所有post使用bert进行编码
    [
        {
            topic: topic_sent,
            post : post_sent,
            enc: bert_
        }
    ]
    保存向量矩阵和对应的主题
    """
    data = load_json('hflqa/faq.json')

    posts, topics = [], []
    for topic, qas in data.items():
        for post in qas['post']:
            posts.append(post)
            topics.append(topic)

    encoder = get_model('./output/transformers-merge3-bert-6L')

    encs = encoder.encode(posts, show_progress_bar=True)

    save_json(topics, 'hflqa/topics.json')

    corpus_mat = np.asarray(encs)
    np.save('hflqa/corpus_mat.npy', corpus_mat)


if __name__ == '__main__':
    index_query()