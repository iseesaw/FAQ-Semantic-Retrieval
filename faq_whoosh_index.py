#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-09-28 19:17:04
# @Author  : Kaiyan Zhang (minekaiyan@gmail.com)
# @Link    : https://github.com/iseesaw
# @Version : 1.0.0
import os
import json
from json import load
from os import write
from tqdm import tqdm
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID

from jieba.analyse import ChineseAnalyzer

source_file = "data/add_faq.json"
stopwords_file = "data/stopwords.txt"
index_dir = "whoosh_index"


def load_data():
    with open(source_file, "r", encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_stopwords():
    stopwords = set([])
    with open(stopwords_file, "r", encoding='utf-8') as f:
        for line in f:
            stopwords.add(line.strip())
    return stopwords


def init():
    data = load_data()
    analyzer = ChineseAnalyzer()
    schema = Schema(pid=ID(stored=True),
                    content=TEXT(stored=True, analyzer=analyzer),
                    topic=ID(stored=True))

    if not os.path.exists(index_dir):
        os.mkdir(index_dir)
    idx = create_in(index_dir, schema)

    writer = idx.writer()
    for i, (topic, _) in enumerate(data.items()):
        writer.add_document(topic=topic, pid="topic-" + str(i), content=topic)

    writer.commit()


if __name__ == '__main__':
    init()