#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-09-02 16:37:30
# @Author  : Kaiyan Zhang (minekaiyan@gmail.com)
# @Link    : https://github.com/iseesaw
# @Version : 1.0.0

import os
import json
import random
from locust import HttpLocust, TaskSet, task, between

def get_seqs():
    seqs = []
    
    return seqs

seqs = get_seqs()

class UserBehaviour(TaskSet):
    # @task
    # def task1(self):
    #     """随机返回若干个问题和id
    #     http://120.25.81.83/psyfaq/init
    #     """
    #     self.client.get("/init")

    @task
    def task2(self):
        """返回相关的若干个问题和id
        http://120.25.81.83/psyfaq/faq
        {"question": "xxx"}
        """
        # json = dict; request.json
        # data = json.dumps; request.data
        q = random.choice(seqs)
        self.client.post("/faq", json={"question": q})

    # @task
    # def task3(self):
    #     """"返回问题的心理疏导意见
    #     http://120.25.81.83/psyfaq/clickq?qid=1
    #     """
    #     self.client.get("/clickq", params={"qid": random.randint(1, 500)})

class WebsiteUser(HttpLocust):
    task_set = UserBehaviour
    # min/max wait time(s)
    wait_time = between(2, 5)

# locust -f load_test.py --host=http://120.25.81.83/psyfaq
# locust  -f load_test.py  --host=http://120.25.81.83/psyfaq --no-web -c 1000 -r 100 -t 3m
