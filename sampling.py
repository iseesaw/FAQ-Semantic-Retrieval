#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-08-28 16:58:56
# @Author  : Kaiyan Zhang (minekaiyan@gmail.com)
# @Link    : https://github.com/iseesaw
# @Version : 1.0.0
import ast
import logging
import argparse

import os
import json
import random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

from sentence_transformers import models, SentenceTransformer
"""
FAQ 负采样, 原始数据保存格式为
{
    topic1: {
        post: [q1, q2, q3],
        resp: [r1, r2, r3, r4]
    },
    topic2: {
        post: [...],
        resp: [...]
    },
    ...
}
其中 topic 下保存相同“主题”的所有问答对
post 保存的都是语义一致的问题（不同表达形式, 可以互为正样本）
resp 保存的是该 topic 的所有可能回复（可以随机选择一个作为回复）

FAQ 的过程为
1. 用户输入 query, 与所有 post 进行相似度计算
2. 确定与 query 最相似的 post 所属的 topic
3. 在确定 topic 的 resp 中随机选择一个作为回复
"""

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO)
logger = logging.getLogger(__name__)


def positive_sampling(data, num_pos=4):
    """在每个主题内进行正采样

    Args:
        data (Dict(str, Dict(str, List[str]))): 
            json形式原始数据
        num_pos (int, optional): 
            正采样个数. Defaults to 4.
            需要依据局部负采样的个数进行调节
            如果局部负采样的个数较多，采集的负样本会偏少
            为了保证最终正负样本比例基本为1:1
            此时正采样个数应该小于全部负采样个数（局部+全局）

    Returns:
        pos_pairs (List[Tuple(str, str)]): 正样本对
    """
    pos_pairs = []
    for _, post_resp in data.items():
        post = post_resp['post']
        total = len(post)
        for i, p in enumerate(post):
            if i < total - 2:
                cands = post[i + 1:]
                cands = cands if 0 < len(cands) <= num_pos else random.sample(
                    cands, k=num_pos)
                pos_pairs.extend([[p, cand, 1] for cand in cands])

    return pos_pairs


def negative_sampling(vectors,
                      sentences,
                      labels,
                      n_clusters,
                      local_num_negs=2,
                      global_num_negs=2,
                      algorithm='kmeans'):
    """负采样
    Step1: 读取所有 topic 的 post 并使用预训练的模型编码得到句向量 vectors
    Step2: 使用 KMeans 对句向量 vectors 进行聚类
            - 理论上聚为 n_topics 类, 每一簇都是相同 topic 的 post
            - 为了确保每一簇类包含语义相似但不属于同一个 topic 的 post
            - 可以聚为 n_topics/m 簇, 其中 m=2,3,4 可调节
    Step3: 根据聚类结果, 进行负采样
            - 局部负采样, 对于每个 post 在每个聚类簇中采样不同 topic 的 post
            - 全局负采样, 对于每个 post 在所有聚类簇中采样不同 topic 的 post
    
    Args:
        vectors (List[List[int]]): 待采样问题句向量
        sentences (List[List[str]]): 待采样问题文本
        labels (List[int]): 待采样问题真实标签（对应的主题，相同标签的问题表达形式不同，但意思相同）
        n_clusters (int): 聚类数，可以指定，也可以自动计算，一般为 total_topics / 2（不能太大或者太小）
        local_num_negs (int, optional): 局部负采样数. Defaults to 2.
        global_num_negs (int, optional): 全局负采样数. Defaults to 2.
        algorithm (str, optional): 聚类算法，kmeans 比 gmm 快很多. Defaults to 'kmeans'.

    Raises:
        NotImplemented: 未实现的聚类方法

    Returns:
        preds (List[int]): 聚类后标签
        neg_pairs (List[Tuple(str, str)]): 负样本对
    """
    assert len(vectors) == len(sentences) == len(labels)
    # ref https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    if algorithm == 'kmeans':
        kmeans = KMeans(n_clusters=n_clusters,
                        random_state=128,
                        init='k-means++',
                        n_init=1,
                        verbose=True)
        kmeans.fit(vectors)
        preds = kmeans.labels_
    elif algorithm == 'gmm':
        model = GaussianMixture(n_components=n_clusters,
                                random_state=128,
                                verbose=2,
                                covariance_type='full')
        model.fit(vectors)
        preds = model.predict(vectors)
    else:
        raise NotImplemented

    pred2sents = {n: [] for n in range(n_clusters)}
    for idx, pred in enumerate(preds):
        sents = pred2sents.get(pred)
        sents.append(idx)

    # 负采样
    neg_pairs = []
    for idx, (label, pred, sent) in enumerate(zip(labels, preds, sentences)):
        # 簇内局部负采样
        cands = [i for i in pred2sents.get(pred) if labels[i] != label]
        num_cands = len(cands)
        if not num_cands:
            pairs = []
        elif num_cands <= local_num_negs:
            pairs = cands
        else:
            pairs = random.sample(cands, k=local_num_negs)

        # 全局负采样
        cand_labels = list(range(n_clusters))
        cand_labels.remove(pred)
        for cand_label in random.sample(cand_labels, k=global_num_negs):
            cands = [
                i for i in pred2sents.get(cand_label) if labels[i] != label
            ]
            if len(cands):
                pairs.append(random.choice(cands))

        neg_pairs.extend([[sent, sentences[pair], 0] for pair in pairs])

    return preds, neg_pairs


def visualize(vectors, labels, preds, output_dir):
    """聚类结果可视化
    （类别太多，基本看不出来效果）

    Args:
        vectors (List[List[int]]): 句向量
        labels (List[int]): 真实标签
        preds (List[int]): 预测标签
        output_dir ([str): 可视化图片保存位置
    """
    # PCA
    pca = PCA(n_components=2, random_state=128)
    pos = pca.fit(vectors).transform(vectors)

    plt.figure(figsize=(12, 12))
    plt.subplot(221)
    plt.title('original label')
    plt.scatter(pos[:, 0], pos[:, 1], c=labels)

    plt.subplot(222)
    plt.title('cluster label')
    plt.scatter(pos[:, 0], pos[:, 1], c=preds)

    plt.savefig(os.path.join(output_dir, 'cluster.png'))


def encode(sentences, model_name_or_path, is_transformers):
    """句向量编码

    Args:
        sentences (List[List[str]]): 待编码句子
        model_name_or_path (str): 预训练模型位置
        is_transformers (bool): 
            是否为 Transformers 模型
            Transformeres 模型需要额外加入平均池化层

    Returns:
        vectors (List[List[int]]): 句向量
    """
    # 使用 BERT 作为 encoder, 并加载预训练模型
    word_embedding_model = models.BERT(model_name_or_path)

    # 使用 mean pooling 获得句向量表示
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False)

    model = SentenceTransformer(
        modules=[word_embedding_model, pooling_model])

    vectors = model.encode(sentences, show_progress_bar=True, device='cuda')

    return vectors


def load_data(filename):
    """读取 FAQ 数据集
    :param filename
    :return
        Dict(str, Dict(str, List[str]))
    """
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def main(args):
    if not os.path.exists(args.output_dir):
        logger.info('make directory %s' % args.output_dir)
        os.makedirs(args.output_dir)

    logger.info('loading data from %s' % args.filename)
    data = load_data(args.filename)

    # 正采样
    logger.info('** positive sampling **')
    logger.info('num_pos = %d' % args.num_pos)
    pos_pairs = positive_sampling(data, num_pos=args.num_pos)
    logger.info('sampling %d positive samples' % len(pos_pairs))

    # 负采样准备（句向量编码）
    logger.info('** negative sampling **')
    sents, labels = [], []
    cnt = 0
    for _, post_resp in data.items():
        sents.extend(post_resp.get('post'))
        labels.extend([cnt] * len(post_resp.get('post')))
        cnt += 1

    logger.info('loading checkpoint from %s' % args.model_name_or_path)
    logger.info('encoding %d sentences' % len(sents))
    vectors = encode(sents,
                     model_name_or_path=args.model_name_or_path,
                     is_transformers=args.is_transformers)
    n_clusters = args.n_clusters if args.n_clusters != -1 else int(
        len(data) // args.hyper_beta)

    logger.info('n_cluster = %d, local_num_negs = %d, global_num_negs = %d' %
                (n_clusters, args.local_num_negs, args.global_num_negs))
    preds, neg_pairs = negative_sampling(vectors,
                                         sentences=sents,
                                         labels=labels,
                                         n_clusters=n_clusters,
                                         local_num_negs=args.local_num_negs,
                                         global_num_negs=args.global_num_negs,
                                         algorithm=args.algorithm)
    logger.info('sampling %d negative samples' % len(neg_pairs))

    # 可视化
    if args.visualized:
        logger.info('** visualize **')
        visualize(vectors,
                  labels=labels,
                  preds=preds,
                  output_dir=args.output_dir)

    # merge & shuffle
    all_pairs = pos_pairs + neg_pairs
    random.shuffle(all_pairs)
    logger.info(
        'we get total %d samples, where %d positive samples and %d negative samples'
        % (len(all_pairs), len(pos_pairs), len(neg_pairs)))

    # split & save
    out_file = f'beta{args.hyper_beta}_{args.algorithm}_p{args.num_pos}_n{args.local_num_negs}{args.global_num_negs}.csv'
    df = pd.DataFrame(data=all_pairs,
                      columns=['sentence1', 'sentence2', 'label'])

    if args.is_split:
        logger.info('train/test set split with test_size = %f' %
                    args.test_size)
        trainset, testset = train_test_split(df, test_size=args.test_size)

        logger.info('save samples to all/train/test_%s' % out_file)
        df.to_csv(os.path.join(args.output_dir, 'all_' + out_file), index=None)
        trainset.to_csv(os.path.join(args.output_dir, 'train_' + out_file),
                        index=None)
        testset.to_csv(os.path.join(args.output_dir, 'test_' + out_file),
                       index=None)
    else:
        logger.info('save all samples to %s' % out_file)
        df.to_csv(os.path.join(args.output_dir, out_file), index=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Positive & Negative Sampling System')

    parser.add_argument(
        '--filename',
        default='ddqa/train_faq.json',
        help='json format original data, like {topic: {post:[], resp:[]}}')
    parser.add_argument(
        '--model_name_or_path',
        default='./output/training-OnlineConstrativeLoss-merge-bert-6L',
        help='path of pretrained model which is used to get sentence vector')
    parser.add_argument(
        '--hyper_beta',
        type=int,
        default=2,
        help='hyperparameter (n_clusters = total_topics / hyper_beta)')
    parser.add_argument(
        '--n_clusters',
        type=int,
        default=-1,
        help='if n_clusters=-1, then n_cluster=n_topics/hyper_m')
    parser.add_argument('--algorithm',
                        type=str,
                        default='kmeans',
                        choices=['kmeans', 'gmm'],
                        help='cluster algorithm')
    parser.add_argument('--num_pos',
                        type=int,
                        default=5,
                        help='number of positive sampling')
    parser.add_argument('--local_num_negs',
                        type=int,
                        default=3,
                        help='number of local negative sampling')
    parser.add_argument('--global_num_negs',
                        type=int,
                        default=2,
                        help='number of global negative sampling')
    parser.add_argument('--visualized',
                        type=ast.literal_eval,
                        default=False,
                        help='whether to visualize cluster results or not')
    parser.add_argument('--is_split',
                        type=ast.literal_eval,
                        default=False,
                        help='whether to split the data into train and test')
    parser.add_argument('--test_size',
                        type=float,
                        default=0.1,
                        help='train/test split size')
    parser.add_argument('--output_dir',
                        type=str,
                        default='./samples',
                        help='directory to save train/test.csv')

    args = parser.parse_args()
    main(args)