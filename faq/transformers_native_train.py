#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-08-29 01:51:01
# @Author  : Kaiyan Zhang (minekaiyan@gmail.com)
# @Link    : https://github.com/iseesaw
# @Version : 1.0.0
import pandas as pd

import torch
from torch.nn import functional as F

from transformers import AutoModel, AutoTokenizer, \
    TrainingArguments, Trainer, AdamW

'''
简易版本，可以参考 sentence-transformers 的实现
https://github.com/UKPLab/sentence-transformers/blob/cfd4e3d4d4ac38f2d06438af783f36c94a571bd1/sentence_transformers/losses/ContrastiveLoss.py#L53
以及 Trainer 的实现
https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer
'''

def train(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModel.from_pretrained(args.model_name_or_path)

    optimizer = AdamW(model.parameters(), lr=1e-5)

    sentences1, sentences2, labels = [], [], []
    s1_encoding = tokenizer(sentences1,
                            return_tensors='pt',
                            padding=True,
                            truncation=True)
    s2_encoding = tokenizer(sentences2,
                            return_tensors='pt',
                            padding=True,
                            truncation=True)
    labels = torch.tensor([1, 0]).unsqueeze(0)

    outputs1 = model(**s1_encoding)
    outputs2 = model(**s2_encoding)

    loss = F.cross_entropy(labels, F.cosine_similarity(outputs1[1],
                                                       outputs2[1]))
    loss.backward()
    optimizer.step()

if __name__ == '__main__':
    train(args)
