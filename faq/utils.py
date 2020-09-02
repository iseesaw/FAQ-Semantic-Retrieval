#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-08-26 01:27:12
# @Author  : Kaiyan Zhang (minekaiyan@gmail.com)
# @Link    : https://github.com/iseesaw
# @Version : 1.0.0
import json
import numpy as np
import pandas as pd
from sentence_transformers import models, SentenceTransformer


# Cosine Similarity
def cos_sim(query_vec, corpus_mat, corpus_norm_mat=None):
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


# Model
def get_model(model_name_or_path, device='cuda'):
    """初始化 SentenceTransformer 编码器

    Args:
        model_name_or_path (str): Transformers 或者微调后的 BERT 模型
        device (str, optional): cpu or cuda. Defaults to 'cuda'.

    Returns:
        SentenceTransformers: 编码器
    """
    word_embedding_model = models.BERT(model_name_or_path)

    # 使用 mean pooling 获得句向量表示
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model],
                                device=device)

    return model


# IO
def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
