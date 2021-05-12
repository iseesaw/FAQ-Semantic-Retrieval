#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-09-05 23:40:14
# @Author  : Kaiyan Zhang (minekaiyan@gmail.com)
# @Link    : https://github.com/iseesaw
# @Version : 1.0.0
import os
import time
import random
import logging
from typing import Optional, Dict
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from functools import lru_cache
from annoy import AnnoyIndex

from utils import load_json
from transformers_encoder import TransformersEncoder

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
FEAT_DIM = 768
TOPK = 10

prefix = "/data/benben/data/faq_bert/"
# prefix = "/users6/kyzhang/benben/FAQ-Semantic-Retrieval"
MODEL_NAME_OR_PATH = os.path.join(prefix, "output/transformers-merge3-bert-6L")
FAQ_FILE = os.path.join(prefix, "ext_hflqa/clean_faq.json")
ANNOY_INDEX_FILE = os.path.join(prefix, "ext_hflqa/index.ann")
IDX2TOPIC_FILE = os.path.join(prefix, "ext_hflqa/idx2topic.json")
VEC_FILE = os.path.join(prefix, "ext_hflqa/vec.npy")

app = FastAPI()

logger.info('加载数据并初始化模型')

logger.info("加载FAQ源文件")
faq = load_json(FAQ_FILE)
idx2topic = load_json(IDX2TOPIC_FILE)
vectors = np.load(VEC_FILE)

logger.info("加载Annoy索引文件")
index = AnnoyIndex(FEAT_DIM, metric='angular')
index.load(ANNOY_INDEX_FILE)

logger.info("加载BERT预训练模型")
encoder = TransformersEncoder(model_name_or_path=MODEL_NAME_OR_PATH)
logger.info('初始化结束')


class User(BaseModel):
    id: str = ''


class LTP(BaseModel):
    seg: str = ''
    arc: str = ''
    ner: str = ''
    pos: str = ''


class AnaphoraResolution(BaseModel):
    score: int = 0
    result: str = ''


class MetaField(BaseModel):
    emotion: str = None
    consumption_class: int = 0
    consumption_result: float = 0.0
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
    vector = np.squeeze(encoder.encode([query]), axis=0)
    res = index.get_nns_by_vector(vector,
                                  TOPK,
                                  search_k=-1,
                                  include_distances=True)
    topic = idx2topic[str(res[0][0] - 1)]["topic"]
    return topic, 1 - res[1][0]


@app.post("/module/FAQ")
def read_item(query: Query):
    st = time.time()
    topic, score = get_res(query.content)
    responses = faq[topic]['resp']
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