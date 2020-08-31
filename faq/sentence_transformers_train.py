#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-08-28 23:15:32
# @Author  : Kaiyan Zhang (minekaiyan@gmail.com)
# @Link    : https://github.com/iseesaw
# @Version : 1.0.0
import os
import pprint
import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader

from sentence_transformers import losses, models
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer, evaluation
from sentence_transformers.readers import InputExample

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


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
    # 使用 BERT 作为 encoder
    word_embedding_model = models.BERT(args.model_name_or_path)

    # 使用 mean pooling 获得句向量表示
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model],
                                device='cuda')

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
    dev_sentences1, dev_sentences2, dev_labels = load_dev_sentences(
        args.devset_path)

    binary_acc_evaluator = evaluation.BinaryClassificationEvaluator(
        dev_sentences1, dev_sentences2, dev_labels)

    # 模型训练
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=binary_acc_evaluator,
              epochs=args.num_epochs,
              warmup_steps=args.warmup_steps,
              output_path=args.model_save_path,
              output_path_ignore_not_empty=True)


def test(args):
    model = SentenceTransformer(args.model_save_path, device='cuda')

    # 开放集评估
    dev_sentences1, dev_sentences2, dev_labels = load_dev_sentences(
        args.devset_path)
    binary_acc_evaluator = evaluation.BinaryClassificationEvaluator(
        dev_sentences1, dev_sentences2, dev_labels)
    model.evaluate(binary_acc_evaluator, args.model_save_path)

    # 开发集阈值
    result = pd.read_csv(
        os.path.join(args.model_save_path,
                     'binary_classification_evaluation_results.csv'))
    max_idx = result['cosine_acc'].argmax()
    threshold = result['cosine_acc_threshold'].values[max_idx]

    # 测试集评估
    sents1, sents2, labels = load_dev_sentences(args.testset_path)
    vec_sents1 = model.encode(sents1,
                              batch_size=args.eval_batch_size,
                              show_progress_bar=True,
                              convert_to_tensor=True)
    vec_sents2 = model.encode(sents2,
                              batch_size=args.eval_batch_size,
                              show_progress_bar=True,
                              convert_to_tensor=True)

    cos = torch.nn.CosineSimilarity()
    scores = cos(vec_sents1, vec_sents2).cpu()

    # 测试集结果
    preds = [1 if s > threshold else 0 for s in scores]
    acc = accuracy_score(labels, preds)
    p_r_f1 = precision_recall_fscore_support(labels, preds, average='macro')
    test_result = {
        'accuracy': acc,
        'macro_precision': p_r_f1[0],
        'macro_recall': p_r_f1[1],
        'macro_f1': p_r_f1[2]
    }
    pprint.pprint(test_result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Sentence Transformers Training.')

    parser.add_argument('--mode',
                        type=str,
                        default='test',
                        help='train or test')
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        default=
        'output/training-OnlineConstrativeLoss-LCQMC-bert-base-chinese/0_BERT'
        #'/users6/kyzhang/embeddings/bert/bert-base-chinese'
        #'/users6/kyzhang/embeddings/distilbert/distilbert-multilingual-nli-stsb-quora-ranking/'
    )
    parser.add_argument('--trainset_path',
                        type=str,
                        default='samples/train.csv')
    parser.add_argument('--devset_path',
                        type=str,
                        default='samples/test.csv')
    parser.add_argument('--testset_path',
                        type=str,
                        default='samples/test.csv')

    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--eval_batch_size', type=int, default=128)

    parser.add_argument('--warmup_steps', type=int, default=1000)

    parser.add_argument(
        '--margin',
        type=float,
        default=0.5,
        help='Negative pairs should have a distance of at least 0.5')
    parser.add_argument(
        '--model_save_path',
        type=str,
        default='output/training-OnlineConstrativeLoss-customized-bert-base-chinese'
    )

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
