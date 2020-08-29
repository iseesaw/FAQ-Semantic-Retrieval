# Lab

> 默认使用 余弦相似度，个别数据上欧式距离和马哈顿距离可能好一些

- Sentence-Transformers + LCQMC train

  - `distilbert-multilingual-nli-stsb-quora-ranking`

  - `training-OnlineConstrativeLoss-LCQMC`

  - 使用 cosine_similarity

  - LCQMC dev 确定阈值

    | dataset | threshold | accuracy | macro f1 |
    | ------- | --------- | -------- | -------- |
    | dev     | 0.8598    | 0.8741   | 0.8753   |
    | test    | -         | 0.8745   | 0.8744   |

    

- Sentence-Transformers + LCQMC train

  - `bert-base-chinese`

  - `training-OnlineConstrativeLoss-LCQMC-bert-base-chinese`

    | dataset | threshold | accuracy | macro f1 |
    | ------- | --------- | -------- | -------- |
    | dev     | 0.8464    | 0.8890   | 0.8898   |
    | test    | -         | 0.8796   | 0.8793   |

    



- Transformers + LCQMC train

  - customized Siamese Network

  - `bert-base-chinese`

  - `transformers-bert-base-chinese`

    | dataset | threshold | accuracy | macro f1 |
    | ------- | --------- | -------- | -------- |
    | dev     |           |          |          |
    | test    |           |          |          |

  - BertForSeqClassification

  - `bert-base-chinese`