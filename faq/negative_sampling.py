#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-08-29 02:39:06
# @Author  : Kaiyan Zhang (minekaiyan@gmail.com)
# @Link    : https://github.com/iseesaw
# @Version : 1.0.0

import os
import numpy as np
from sklearn.cluster import KMeans


def encode():
    '''对所有样本进行向量表示
    '''


def cluster():
    '''进行 Kmeans 聚类
    参考文档聚类 https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html#sphx-glr-auto-examples-text-plot-document-clustering-py
    '''
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    kmeans.labels_
    #array([1, 1, 1, 0, 0, 0], dtype=int32)
    kmeans.predict([[0, 0], [12, 3]])
    #array([1, 0], dtype=int32)
    kmeans.cluster_centers_
    #array([[10.,  2.],
    #    [ 1.,  2.]])


def sampling():
    '''对于非同类样本进行负采样
    '''


if __name__ == '__main__':
    pass