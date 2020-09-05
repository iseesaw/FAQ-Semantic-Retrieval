#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-09-03 04:00:10
# @Author  : Kaiyan Zhang (minekaiyan@gmail.com)
# @Link    : https://github.com/iseesaw
# @Version : 1.0.0
from transformers_encoder import TransformersEncoder
import time
import random
import logging
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from functools import lru_cache

from utils import load_json, cos_sim

model_name_or_path = 'transformers-merge3-bert-6L'
faq_file = 'hflqa/faq.json'
corpus_mat_file = 'hflqa/corpus_mat.npy'
topics_file = 'hflqa/topics.json'

app = FastAPI()


def init_data():
    """加载数据
    """
    faq_data = load_json(faq_file)
    corpus_mat = np.load(corpus_mat_file)
    topics = load_json(topics_file)
    corpus_mat_norm = np.linalg.norm(corpus_mat, axis=1)
    return faq_data, topics, corpus_mat, corpus_mat_norm


print('start loading')
faq_data, topics, corpus_mat, corpus_mat_norm = init_data()
encoder = TransformersEncoder(model_name_or_path=model_name_or_path)
print('end loading...')


class Query(BaseModel):
    query: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@lru_cache(maxsize=512)
def get_res(query):
    enc = encoder.encode([query])
    scores = cos_sim(np.squeeze(enc, axis=0), corpus_mat, corpus_mat_norm)
    max_index = np.argmax(scores)
    topic = topics[max_index]
    return topic


@app.post("/module/ext_faq_test")
def read_item(query: Query):
    st = time.time()
    topic = get_res(query.query)
    responses = faq_data[topic]['resp']
    reply = random.choice(responses)
    print('--------------')
    print('Query:', query)
    print('Reply:', reply)
    print('Takes %f' % (time.time() - st))
    return {'reply': reply}


# uvicorn faq_app_fastapi:app --reload --port 8889