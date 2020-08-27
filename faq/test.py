#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-08-26 01:46:35
# @Author  : Kaiyan Zhang (minekaiyan@gmail.com)
# @Link    : https://github.com/iseesaw
# @Version : 1.0.0
import time

import torch
import torch.nn as nn

import pandas as pd
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity

from utils import load_json, cos_dist
from enc_client import EncodeClient


def dis_test():
    '''BERT embedding 计算相似度
    '''
    client = EncodeClient()
    df = pd.read_csv('lcqmc/LCQMC_dev.csv')
    sent1_loader = tqdm(DataLoader(df['sentence1'][:3],
                                   batch_size=1,
                                   shuffle=False),
                        desc='Iteration')
    sent2_loader = DataLoader(df['sentence2'][:3], batch_size=1, shuffle=False)

    cos = nn.CosineSimilarity()
    for sent1_batch, sent2_batch in zip(sent1_loader, sent2_loader):
        sent1_enc = client.encode(sent1_batch)
        sent2_enc = client.encode(sent2_batch)
        score = cos(torch.tensor(sent1_enc), torch.tensor(sent2_enc)).tolist()
        print('------')
        print(sent1_batch, sent2_batch)
        print(score)


def construct_pos():
    '''根据faq数据构造正例
    '''
    data = load_json('hflqa/faq.json')
    topics, ques, ans = [], [], []
    for topic, qas in data.items():
        for q in qas['post']:
            for a in qas['resp']:
                ques.append(q)
                ans.append(a)
                topics.append(topic)

    df = pd.DataFrame(data={
        'topic': topics,
        'query': ques,
        'answer': ans
    },
                      columns=['topic', 'query', 'answer'])

    df.to_csv('pos.csv', index=None)

    print(df.shape)
    # (339886, 3)


def cost_test():
    '''测试各种余弦距离计算函数耗时
    自定义函数，可以提前计算相关值，速度最快

    num_cands   20000   100000     
    sklearn    0.2950s  1.3517s
    torch      0.1851s  0.9408s
    custom     0.0092s  0.0673s
    '''
    post = np.random.randn(100000, 768)
    query = np.random.randn(1, 768)
    post_norm = np.linalg.norm(post)

    cos = nn.CosineSimilarity()

    print('---- sklearn ----')
    t1 = time.time()
    scores = cosine_similarity(query, post)
    print(np.argmax(scores))
    t2 = time.time()
    print(t2 - t1)

    print('---- torch ----')
    scores = cos(torch.tensor(query), torch.tensor(post)).tolist()
    print(np.argmax(scores))
    t3 = time.time()
    print(t3 - t2)

    print('---- custom ----')
    scores = cos_dist(np.squeeze(query, axis=0), post, post_norm)
    print(np.argmax(scores))
    t4 = time.time()
    print(t4 - t3)

if __name__ == '__main__':
    dis_test()