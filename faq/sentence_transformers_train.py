#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-08-28 23:15:32
# @Author  : Kaiyan Zhang (minekaiyan@gmail.com)
# @Link    : https://github.com/iseesaw
# @Version : 1.0.0

import logging
import argparse

import pandas as pd
from datetime import datetime

from torch.utils.data import DataLoader
from sentence_transformers import losses, util
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer, evaluation
from sentence_transformers.readers import InputExample

# 打印日志
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


def load_train_samples(filename):
    data = pd.read_csv(filename)
    samples = []
    for _, row in data.iterrows():
        samples.append(
            InputExample(texts=[row['sentence1'], row['sentence2']],
                         label=int(row['label'])))
    return samples


def load_dev_sentences(filename):
    data = pd.read_csv(filename)
    sents1, sents2, labels = [], [], []
    for _, row in data.iterrows():
        sents1.append(row['sentence1'])
        sents2.append(row['sentence2'])
        labels.append(int(row['label']))
    return sents1, sents2, labels


def train(args):
    # 初始化基础模型
    model = SentenceTransformer(args.model_name_or_path, device='cuda')

    # 使用余弦距离作为度量指标 (cosine_distance = 1-cosine_similarity)
    distance_metric = losses.SiameseDistanceMetric.COSINE_DISTANCE

    # 读取训练集
    train_samples = load_train_samples(args.trainset_path)
    train_dataset = SentencesDataset(train_samples, model=model)
    train_dataloader = DataLoader(train_dataset,
                                  shuffle=True,
                                  batch_size=args.train_batch_size)
    # 初始化训练损失函数
    train_loss = losses.OnlineContrastiveLoss(model=model,
                                              distance_metric=distance_metric,
                                              margin=args.margin)

    # 构造开发集评估器
    # 给定 (sentence1, sentence2) 判断是否相似
    # 评估器将计算两个句向量的余弦相似度，如果高于某个阈值则判断为相似
    dev_sentences1, dev_sentences2, dev_labels = load_dev_sentences(args.devset_path)

    binary_acc_evaluator = evaluation.BinaryClassificationEvaluator(
        dev_sentences1, dev_sentences2, dev_labels)

    # 模型训练
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=binary_acc_evaluator,
              epochs=args.num_epochs,
              warmup_steps=args.warmup_steps,
              output_path=args.model_save_path,
              output_path_ignore_not_empty=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Sentence Transformers Training.')

    parser.add_argument(
        '--model_name_or_path',
        type=str,
        default=
        '/users6/kyzhang/embeddings/distilbert/distilbert-multilingual-nli-stsb-quora-ranking/'
    )
    parser.add_argument('--trainset_path',
                        type=str,
                        default='lcqmc/LCQMC_train.csv')
    parser.add_argument('--devset_path',
                        type=str,
                        default='lcqmc/LCQMC_dev.csv')

    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--warmup_steps', type=int, default=1000)

    parser.add_argument(
        '--margin',
        type=float,
        default=0.5,
        help='Negative pairs should have a distance of at least 0.5')
    parser.add_argument('--model_save_path',
                        type=str,
                        default='output/training-OnlineConstrativeLoss-LCQMC')

    args = parser.parse_args()
    train(args)