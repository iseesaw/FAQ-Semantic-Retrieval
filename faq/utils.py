#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-08-26 01:27:12
# @Author  : Kaiyan Zhang (minekaiyan@gmail.com)
# @Link    : https://github.com/iseesaw
# @Version : 1.0.0
import json
import numpy as np
import requests

BERT_URL = 'http://127.0.0.1:8125/encode'
headers = {"Content-Type": "application/json"}


def bert_encode(query):
    post_data = {"id": query, "texts": [query], "is_tokenized": False}
    try:
        r = requests.post(BERT_URL,
                          data=json.dumps(post_data),
                          headers=headers,
                          timeout=5)
        res = r.json()
        enc = None
        if res['status'] == 200 and res['id'] == post_data['id']:
            results = res['result']
            if results:
                enc = np.array(results[0])
    except Exception as e:
        print(e)
        return None
    return enc


def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_npy(filename):
    with open(filename, 'rb') as f:
        return np.load(f)


def save_npy(data, filename):
    with open(filename, 'wb') as f:
        np.save(f, data)


def cos_dist(query_vec, corpus_mat, corpus_norm_mat=None):
    '''余弦相似度计算
    
    :param query_vec: ndarray, (dim_size)
    :param corpus_mat: ndarray, (num_cands, dim_size)
    :param corpus_norm_mat: ndarray, (num_cands) 可提前计算加快速度
    :return: ndarray, (num_cands)
    '''
    if not corpus_norm_mat:
        corpus_norm_mat = np.linalg.norm(corpus_mat)
    return np.dot(corpus_mat,
                  query_vec) / (np.linalg.norm(query_vec) * corpus_norm_mat)
