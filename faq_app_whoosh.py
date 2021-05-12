#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-09-28 19:16:29
# @Author  : Kaiyan Zhang (minekaiyan@gmail.com)
# @Link    : https://github.com/iseesaw
# @Version : 1.0.0

import time
import random
import logging
import numpy as np
from typing import Optional, Dict
from fastapi import FastAPI
from pydantic import BaseModel
from functools import lru_cache

from whoosh.query import Term
from whoosh.qparser import QueryParser, MultifieldParser
from whoosh.index import open_dir

from faq_whoosh_index import load_data, load_stopwords
source_file = "data/add_faq.json"
app = FastAPI()

faq_data = load_data()

IDX = open_dir(dirname="whoosh_index")

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
    try:
        with IDX.searcher() as searcher:
            parser = QueryParser("content", schema=IDX.schema)
            q = parser.parse(query)
            results = searcher.search(q)
            topic = results[0]["topic"]
            return topic, results[0].score
    except Exception:
        return None, 0.


@app.post("/module/ext_faq")
def read_item(query: Query):
    st = time.time()
    topic, score = get_res(query.content)
    if topic:
        responses = faq_data[topic]
        reply = random.choice(responses)
    else:
        reply = "我有点不太明白呢"

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


# uvicorn faq_app_whoosh:app --reload --port 8889