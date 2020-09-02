#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-09-02 16:37:58
# @Author  : Kaiyan Zhang (minekaiyan@gmail.com)
# @Link    : https://github.com/iseesaw
# @Version : 1.0.0

import time
import random
import logging
import numpy as np
from flask import Flask, request, jsonify
from flask_caching import Cache
from utils import load_json, cos_sim
from transformers_encoder import TransformersEncoder
from bert_serving.client import BertClient

model_name_or_path = 'output/transformers-merge3-bert-6L'
faq_file = 'hflqa/faq.json'
corpus_mat_file = 'hflqa/corpus_mat.npy'
topics_file = 'hflqa/topics.json'

# logging.basicConfig(
#     format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
#     datefmt="%Y/%m/%d %H:%M:%S",
#     level=logging.INFO)
# logger = logging.getLogger(__name__)

app = Flask(__name__)
# https://github.com/sh4nks/flask-caching
cache = Cache(app, config={'CACHE_TYPE': 'simple'})


def init_data():
    """加载数据
    """
    faq_data = load_json(faq_file)
    corpus_mat = np.load(corpus_mat_file)
    topics = load_json(topics_file)
    corpus_mat_norm = np.linalg.norm(corpus_mat)
    return faq_data, topics, corpus_mat, corpus_mat_norm


print('start loading')
faq_data, topics, corpus_mat, corpus_mat_norm = init_data()
# encoder = TransformersEncoder(model_name_or_path=model_name_or_path)
encoder = BertClient()
print('end loading...')


@app.route('/module/ext_faq_test', methods=['POST'])
def query():
    query = request.json.get('query')
    topic = cache.get(query)
    if not topic:
        enc = encoder.encode([query])
        scores = cos_sim(np.squeeze(enc, axis=0), corpus_mat, corpus_mat_norm)
        max_index = np.argmax(scores)
        topic = topics[max_index]
        cache.set(query, topic)

    responses = faq_data[topic]['resp']
    reply = random.choice(responses)
    print('--------------')
    print('Query:', query)
    print('Reply:', reply)
    return jsonify({'reply': reply})


if __name__ == '__main__':
    # gunicorn -k eventlet -w 1 -b 127.0.0.1:8889 faq_app:app
    app.run(host='127.0.0.1', port=11122)