#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-08-26 01:27:12
# @Author  : Kaiyan Zhang (minekaiyan@gmail.com)
# @Link    : https://github.com/iseesaw
# @Version : 1.0.0
import json
import numpy as np
import pandas as pd


# Cosine Similarity
def cos_sim(query_vec, corpus_mat, corpus_norm_mat=None):
    '''余弦相似度计算
    
    :param query_vec: ndarray, (dim_size)
    :param corpus_mat: ndarray, (num_cands, dim_size)
    :param corpus_norm_mat: ndarray, (num_cands) 可提前计算加快速度
    :return: ndarray, (num_cands)
    '''
    if corpus_norm_mat is None:
        corpus_norm_mat = np.linalg.norm(corpus_mat)
    return np.dot(corpus_mat,
                  query_vec) / (np.linalg.norm(query_vec) * corpus_norm_mat)


# IO
def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
