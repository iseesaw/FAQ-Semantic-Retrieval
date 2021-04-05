#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-08-26 01:27:04
# @Author  : Kaiyan Zhang (minekaiyan@gmail.com)
# @Link    : https://github.com/iseesaw
# @Version : 1.0.0
import numpy as np

from utils import load_json, save_json
from transformers_encoder import TransformersEncoder

input_faq_file = 'ext_hflqa/clean_faq.json'
output_topic_file = 'ext_hflqa/topics.json'
output_corpus_mat_file = 'ext_hflqa/corpus_mat.npy'
model_path = './output/transformers-merge3-bert-6L'

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
    data = load_json(input_faq_file)

    posts, topics = [], []
    for topic, qas in data.items():
        for post in qas['post']:
            posts.append(post)
            topics.append(topic)

    encoder = TransformersEncoder(model_path)

    encs = encoder.encode(posts, show_progress_bar=True)

    save_json(topics, output_topic_file)

    corpus_mat = np.asarray(encs)
    np.save(output_corpus_mat_file, corpus_mat)


if __name__ == '__main__':
    index_query()
