#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-09-02 16:37:30
# @Author  : Kaiyan Zhang (minekaiyan@gmail.com)
# @Link    : https://github.com/iseesaw
# @Version : 1.0.0

import json
import random
from locust import HttpUser, task, between


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

class FAQUser(HttpUser):
    # min/max wait time
    wait_time = between(2,5)

    @task
    def faq(self):
        self.client.post("/ext_faq_test", json={"query": random.choice(seqs)})

# https://docs.locust.io/en/stable/index.html
# locust  -f locust_test.py  --host=http://127.0.0.1:8889/module --headless -u 1000 -r 100 -t 3m
