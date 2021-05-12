#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-08-26 01:46:35
# @Author  : Kaiyan Zhang (minekaiyan@gmail.com)
# @Link    : https://github.com/iseesaw
# @Version : 1.0.0

import time
import random
import numpy as np
import pandas as pd

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sentence_transformers import SentenceTransformer, util, models

from transformers import BertTokenizer, BertModel

from utils import load_json, cos_sim, save_json


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


def get_faq_corpus_embeddings(embedder, filename='hflqa/faq.json'):
    '''读取 faq 数据并使用 sentence-transformers 进行向量编码
    '''
    data = load_json(filename)
    corpus = []
    for _, post_replys in data.items():
        corpus.extend(post_replys['post'])

    corpus_embeddings = embedder.encode(corpus,
                                        show_progress_bar=True,
                                        convert_to_tensor=True)
    return corpus, corpus_embeddings


def sentence_transformers_test(top_k=5):
    '''使用 sentence-transformers 进行向量编码
    使用 util.pytorch_cos_sim 计算余弦相似度
    使用 np.argpartition 获取 topk
    '''
    embedder = SentenceTransformer('bert-base-chinese')

    corpus, corpus_embeddings = get_faq_corpus_embeddings(embedder)

    while True:
        query = input('Enter: ')
        query_embeddings = embedder.encode([query], convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(query_embeddings,
                                          corpus_embeddings)[0].cpu()

        #We use np.argpartition, to only partially sort the top_k results
        top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]

        print("\n\n======================\n\n")
        print("Query:", query)
        print("\nTop 5 most similar sentences in corpus:")

        for idx in top_results[0:top_k]:
            print(corpus[idx].strip(), "(Score: %.4f)" % (cos_scores[idx]))


def sentence_search_test(topk=5):
    '''使用 sentence-transformers 进行向量编码
    调用 util.sementic_search 进行语义召回检索前 topk
    '''
    embedder = SentenceTransformer(
        './output/training-OnlineConstrativeLoss-LCQMC', device='cuda')

    corpus, corpus_embeddings = get_faq_corpus_embeddings(embedder)
    print("Corpus loaded with {} sentences / embeddings".format(
        len(corpus_embeddings)))

    while True:
        inp_question = input("Please enter a question: ")

        start_time = time.time()
        question_embedding = embedder.encode(inp_question,
                                             convert_to_tensor=True)
        # （num_query, num_corpus）
        hits = util.semantic_search(question_embedding, corpus_embeddings)
        end_time = time.time()

        # Get the hits for the first query
        hits = hits[0]

        print("Input question:", inp_question)
        print("Results (after {:.3f} seconds):".format(end_time - start_time))
        for hit in hits[:topk]:
            print("\t{:.3f}\t{}".format(hit['score'],
                                        corpus[hit['corpus_id']]))

        print("\n\n========\n")


def load_ddqa():
    # v_inc_sim_q_id,std_q_id,std_q,similar_q,tags,src,start_date,end_date,rank
    data = pd.read_csv('ddqa/faq.csv')
    ques, labels = [], []
    for _, row in data.iterrows():
        if row['rank'] == 1:
            labels.append(row['std_q_id'])
            ques.append(row['std_q'])
        labels.append(row['std_q_id'])
        ques.append(row['similar_q'])

    return ques, labels


def load_faq(filename='hflqa/faq.json'):
    data = load_json(filename)
    ques, labels = [], []
    for idx, (topic, post_resp) in enumerate(data.items()):
        for post in post_resp['post']:
            ques.append(post)
            labels.append(topic)

    return ques, labels


def compute_acc():
    is_transformers = True
    # model_path = '/users6/kyzhang/embeddings/bert/bert-base-chinese'
    # model_path = './output/training-OnlineConstrativeLoss-hflqa-beta1.5-gmm-bert/0_BERT'
    # model_path = './output/transformers-merge-bert-base-chinese'
    model_path = './output/transformers-merge3-bert'
    if is_transformers:
        # 使用 BERT 作为 encoder
        word_embedding_model = models.BERT(model_path)
        # 使用 mean pooling 获得句向量表示
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False)
        embedder = SentenceTransformer(
            modules=[word_embedding_model, pooling_model], device='cuda')
    else:
        embedder = SentenceTransformer(model_path, device='cuda')

    # ques, labels = load_ddqa() #load_hflqa()
    # X_train, X_test, y_train, y_test = train_test_split(ques,
    #                                                     labels,
    #                                                     test_size=0.1)

    proj = 'hflqa'
    X_train, y_train = load_faq(f'{proj}/train_faq.json')
    X_test, y_test = load_faq(f'{proj}/test_faq.json')

    corpus_embeddings = embedder.encode(X_train,
                                        show_progress_bar=True,
                                        batch_size=512,
                                        convert_to_tensor=True)
    # corpus_mat_norm = np.linalg.norm(corpus_embeddings)

    query_embeddings = embedder.encode(X_test,
                                       show_progress_bar=True,
                                       batch_size=512,
                                       convert_to_tensor=True)

    print(query_embeddings.shape, corpus_embeddings.shape)
    hits = util.semantic_search(query_embeddings, corpus_embeddings)
    res = [
        1 if y in [y_train[hit[i]['corpus_id']] for i in range(10)] else 0
        for hit, y in zip(hits, y_test)
    ]
    acc = sum(res) / len(res)
    print(acc)

    # data = []
    # for x, y, hit in zip(X_test, y_test, hits):
    #     cands = [y_train[hit[i]['corpus_id']] for i in range(3)]
    #     if y not in cands:
    #         cands.insert(0, y)
    #         cands.insert(0, x)
    #         data.append(cands)
    # pd.DataFrame(data=data, columns=['query', 'std_q', 'error1', 'error2', 'error3']).to_csv('error_cands.csv', index=None)

    # return acc
    # while True:
    #     inp_question = input("Please enter a question: ")

    #     start_time = time.time()
    #     question_embedding = embedder.encode(inp_question,
    #                                          convert_to_tensor=True)
    #     # （num_query, num_corpus）
    #     hits = util.semantic_search(question_embedding, corpus_embeddings)
    #     end_time = time.time()

    #     # Get the hits for the first query
    #     hits = hits[0]

    #     print("Input question:", inp_question)
    #     print("Results (after {:.3f} seconds):".format(end_time - start_time))
    #     for hit in hits[:5]:
    #         print("\t{:.3f}\t{}".format(hit['score'],
    #                                     qs[hit['corpus_id']]))

    #     print("\n\n========\n")


def split_faq(proj):
    data = load_json(f'{proj}/faq.json')
    topics, posts = [], []
    for topic, post_resp in data.items():
        for post in post_resp['post']:
            topics.append(topic)
            posts.append(post)

    train_posts, test_posts, train_topics, test_topics = train_test_split(
        posts, topics, test_size=0.1)

    save_faq(train_posts, train_topics, f'{proj}/train_faq.json')
    save_faq(test_posts, test_topics, f'{proj}/test_faq.json')


def save_faq(posts, topics, filename):
    data = {}
    for post, topic in zip(posts, topics):
        if topic not in data:
            data[topic] = {'post': []}
        data[topic]['post'].append(post)

    save_json(data, filename)


def merge():
    # for mode in ['train', 'test']:
    #     lcqmc = pd.read_csv(f'lcqmc/LCQMC_{mode}.csv')
    #     hflqa = pd.read_csv(f'samples/{mode}_beta1.5_gmm_p4_n42.csv')

    #     total = shuffle(pd.concat([lcqmc, hflqa]))
    #     total.to_csv(f'samples/merge_{mode}_beta1.5_gmm_p4_n42.csv', index=None)

    lcqmc = pd.read_csv(f'samples/ddqa_beta2_kmeans_p5_n32.csv')
    hflqa = pd.read_csv(f'samples/merge_train_beta1.5_gmm_p4_n42.csv')

    total = shuffle(pd.concat([lcqmc, hflqa]))
    total.to_csv(f'samples/merge3.csv', index=None)


def export_ddqa():
    ques, labels = load_ddqa()
    data = {}
    for l, q in zip(labels, ques):
        if l not in data:
            data[l] = {'post': []}
        data[l]['post'].append(q)

    save_json(data, 'ddqa/faq.json')


def save_pretrained_model():
    model_path = './output/transformers-merge3-bert/'
    model = BertModel.from_pretrained(model_path)

    torch.save(model.state_dict(),
               'output/transformers-merge3-bert-6L/pytorch_model.bin')

def for_index():
    train_faq = load_json('hflqa/test_faq.json')
    faq = load_json('hflqa/faq.json')
    for topic in train_faq:
        train_faq[topic]['resp'] = faq[topic]['resp']
    save_json(train_faq, 'hflqa/test_faq_resp.json')

def req_test():
    import requests
    url = '###'
    data = load_json('hflqa/test_faq_resp.json')
    hits = total = fails = 0
    st = time.time()
    for _, post_resp in tqdm(data.items()):
        resp = set(post_resp.get('resp'))
        for post in post_resp.get('post'):
            try:
                reply = requests.post(url=url, json={'query': post}, timeout=1000)
                if reply.json().get('reply') in resp:
                    hits += 1
            except:
                fails += 1
            finally:
                total += 1
    print(f'hits = {hits}, fails = {fails}, total = {total}, avg.sec = {(time.time()-st)/total}')

def request():
    data = {
        'content': '\u4f60\u8fd8\u597d\u5417',
        'msg_type': 'text',
        'metafield': {
            'emotion': 'sad',
            'consumption_class': 0,
            'ltp': {
                'seg': '\u4f60 \u8fd8 \u597d \u5417',
                'arc': '3:SBV 3:ADV 0:HED 3:RAD',
                'ner': 'O O O O',
                'pos': 'r d a u'
            },
            'consumption_result': 0.0,
            'multi_turn_status_dict': {},
            'anaphora_resolution': {
                'score': 0,
                'result': ''
            }
        },
        'user': {
            'id': 'oFeuIs252VLW7ILAKQ1Rh5JViEks'
        },
        'context': {}
    }
    import requests
    url = 'http://127.0.0.1:12345/module/FAQ'
    res = requests.post(url=url, json=data)
    print(res.json())

import json
def count_faq():
    with open("ext_hflqa/clean_faq.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    print("topic", len(data))
    print("post", sum([len(v["post"]) for k, v in data.items()]))
    print("resp", sum([len(v["resp"]) for k, v in data.items()]))

if __name__ == '__main__':
    # dis_test()
    # sentence_transformers_test()
    # compute_acc()
    # request()
    # for_index()
    count_faq()
    """scores = []
    for _ in range(5):
        scores.append(compute_acc())
    print(scores)
    print(sum(scores)/len(scores))"""
    # split_faq('ddqa')
    # merge()
    # export_ddqa()
    # save_pretrained_model()
    # req_test()
