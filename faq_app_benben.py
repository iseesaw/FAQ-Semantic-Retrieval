#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-09-05 23:40:14
# @Author  : Kaiyan Zhang (minekaiyan@gmail.com)
# @Link    : https://github.com/iseesaw
# @Version : 1.0.0
from transformers_encoder import TransformersEncoder
import time
import random
import logging
from typing import Optional, Dict
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from functools import lru_cache

from utils import load_json, cos_sim

##### 日志配置 #####
logger = logging.getLogger()
logger.setLevel('DEBUG')
BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
chlr = logging.StreamHandler()  # 输出到控制台的handler
chlr.setFormatter(formatter)
chlr.setLevel('INFO')
fhlr = logging.FileHandler('example.log')  # 输出到文件的handler
fhlr.setFormatter(formatter)
logger.addHandler(chlr)
logger.addHandler(fhlr)

##### 模型文件配置 #####
model_name_or_path = 'output/transformers-merge3-bert-6L'
faq_file = 'hflqa/faq.json'
corpus_mat_file = 'hflqa/corpus_mat.npy'
topics_file = 'hflqa/topics.json'

app = FastAPI()


def init_data():
    """加载数据
    通过 faq_index.py 生成
    """
    faq_data = load_json(faq_file)
    corpus_mat = np.load(corpus_mat_file)
    topics = load_json(topics_file)
    corpus_mat_norm = np.linalg.norm(corpus_mat, axis=1)
    return faq_data, topics, corpus_mat, corpus_mat_norm


logger.info('加载数据并初始化模型')
faq_data, topics, corpus_mat, corpus_mat_norm = init_data()
encoder = TransformersEncoder(model_name_or_path=model_name_or_path)
logger.info('初始化结束')


class User(BaseModel):
    id: str


class LTP(BaseModel):
    seg: str
    arc: str
    ner: str
    pos: str


class AnaphoraResolution(BaseModel):
    score: int
    result: str


class MetaField(BaseModel):
    emotion: str
    consumption_class: int
    consumption_result: float
    ltp: LTP
    anaphora_resolution: AnaphoraResolution
    score: Optional[float] = None


class Query(BaseModel):
    content: str
    msg_type: str
    metafield: MetaField
    user: User
    context: Optional[Dict[str, str]] = {}


@app.get("/")
def read_root():
    return {"Hello": "World"}


@lru_cache(maxsize=512)
def get_res(query):
    enc = encoder.encode([query])
    scores = cos_sim(np.squeeze(enc, axis=0), corpus_mat, corpus_mat_norm)
    max_index = np.argmax(scores)
    #topk=5
    #top_results = np.argpartition(-scores, range(topk))[0:topk]
    topic = topics[max_index]
    score = scores[max_index]
    return topic, score.item()


@app.post("/module/FAQ")
def read_item(query: Query):
    st = time.time()
    topic, score = get_res(query.content)
    responses = faq_data[topic]['resp']
    reply = random.choice(responses)
    logger.info('Query: %s' % query.content)
    logger.info('Reply: %s' % reply)
    logger.info('Takes: %.6f sec, Score: %.6f' % (time.time() - st, score))

    metafield = query.metafield
    metafield.score = score
    return {
        'msg_type': 'text',
        'reply': reply,
        'context': {},
        'status': 0,
        'metafield': metafield
    }


# uvicorn faq_app_fastapi:app --reload --port 8889