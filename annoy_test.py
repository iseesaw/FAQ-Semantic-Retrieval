#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-04-05 16:21:32
# @Author  : Kaiyan Zhang (minekaiyan@gmail.com)
# @Link    : https://github.com/iseesaw
# @Version : 1.0.0

from transformers_encoder import TransformersEncoder
import os
import time
import numpy as np
from tqdm import tqdm
from annoy import AnnoyIndex

from utils import load_json, save_json

FEAT_DIM = 768
TOPK = 10

# prefix = "/data/benben/data/faq_bert/"
prefix = "/users6/kyzhang/benben/FAQ-Semantic-Retrieval"
MODEL_NAME_OR_PATH = os.path.join(prefix, "output/transformers-merge3-bert-6L")
FAQ_FILE = os.path.join(prefix, "ext_hflqa/clean_faq.json")
ANNOY_INDEX_FILE = os.path.join(prefix, "ext_hflqa/index.ann")
IDX2TOPIC_FILE = os.path.join(prefix, "ext_hflqa/idx2topic.json")
VEC_FILE = os.path.join(prefix, "ext_hflqa/vec.npy")

faq = load_json(FAQ_FILE)

encoder = TransformersEncoder(model_name_or_path=MODEL_NAME_OR_PATH,
                              batch_size=1024)

####### encode posts
if os.path.exists(IDX2TOPIC_FILE) and os.path.exists(VEC_FILE):
    print("Loading idx2topic and vec...")
    idx2topic = load_json(IDX2TOPIC_FILE)
    vectors = np.load(VEC_FILE)
else:
    idx = 0
    idx2topic = {}
    posts = []
    for topic, post_resp in tqdm(faq.items()):
        for post in post_resp["post"]:
            idx2topic[idx] = {"topic": topic, "post": post}
            posts.append(post)
            idx += 1

    encs = encoder.encode(posts, show_progress_bar=True)

    save_json(idx2topic, IDX2TOPIC_FILE)
    vectors = np.asarray(encs)
    np.save(VEC_FILE, vectors)

####### index and test
index = AnnoyIndex(FEAT_DIM, metric='angular')
if os.path.exists(ANNOY_INDEX_FILE):
    print("Loading Annoy index file")
    index.load(ANNOY_INDEX_FILE)
else:
    # idx2topic = {}
    # idx = 0
    # for topic, post_resp in tqdm(faq.items()):
    #     posts = post_resp["post"]
    #     vectors = encoder.encode(posts)
    #     for post, vector in zip(posts, vectors):
    #         idx2topic[idx] = {"topic": topic, "post": post}
    #         # index.add_item(idx, vector)
    #         idx += 1
    # save_json(idx2topic, IDX2TOPIC_FILE)
    # index.save(ANNOY_INDEX_FILE)

    for idx, vec in tqdm(enumerate(vectors)):
        index.add_item(idx, vec)
    index.build(30)
    index.save(ANNOY_INDEX_FILE)

while True:
    query = input(">>> ")
    st = time.time()
    vector = np.squeeze(encoder.encode([query]), axis=0)
    res = index.get_nns_by_vector(vector,
                                  TOPK,
                                  search_k=-1,
                                  include_distances=True)
    print(time.time() - st)
    print(res)
