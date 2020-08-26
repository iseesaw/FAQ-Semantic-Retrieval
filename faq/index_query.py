#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-08-26 01:27:04
# @Author  : Kaiyan Zhang (minekaiyan@gmail.com)
# @Link    : https://github.com/iseesaw
# @Version : 1.0.0
import torch
from bert_serving.client import BertClient
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import load_json, save_json


def index_query():
    '''对所有post使用bert进行编码
    [
        {
            topic: topic_sent,
            post : post_sent,
            enc: bert_
        }
    ]
    '''
    data = load_json('hflqa/faq.json')

    posts, topics = [], []
    for topic, qas in data.items():
        for post in qas['post']:
            posts.append(post)
            topics.append(topic)

    client = BertClient()
    dataloader = tqdm(DataLoader(posts, batch_size=512, shuffle=False),
                      desc='Iteration')

    encs = []
    for batch in dataloader:
        enc = client.encode(batch)
        encs.extend(enc.tolist())

    enc_data = []
    for t, p, e in zip(topics, posts, encs):
        enc_data.append({'topic': t, 'post': p, 'enc': e})

    save_json(enc_data, 'hflqa/index.json')

if __name__ == '__main__':
    index_query()