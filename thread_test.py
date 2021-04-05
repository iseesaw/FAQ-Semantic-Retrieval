#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-09-03 00:56:04
# @Author  : Kaiyan Zhang (minekaiyan@gmail.com)
# @Link    : https://github.com/iseesaw
# @Version : 1.0.0

from threading import Thread
import requests
import random
import json
import time

API_URL = "http://127.0.0.1:8889/module/ext_faq_test"
NUM_REQUESTS = 500
SLEEP_COUNT = 0.1

def get_seqs():
    """获取用户输入样例

    Returns:
        List[str]
    """
    seqs = []
    with open('./hflqa/test_faq_resp.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    for _, post_resp in data.items():
        seqs.extend(post_resp['post'])
    random.shuffle(seqs)
    return seqs


seqs = get_seqs()

def call_predict_endpoint(n):
    payload = {"query": random.choice(seqs)}

    r = requests.post(API_URL, files=payload).json()

    if r["reply"]:
        print("[INFO] thread {} OK".format(n))

    else:
        print("[INFO] thread {} FAILED".format(n))

for i in range(0, NUM_REQUESTS):
    t = Thread(target=call_predict_endpoint, args=(i,))
    t.daemon = True
    t.start()
    time.sleep(SLEEP_COUNT)

time.sleep(300)