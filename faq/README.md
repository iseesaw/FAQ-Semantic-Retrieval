# README

基于 BERT 索引的 FAQ

TODO：
- 已有的 BERT 编码使用 [bert-as-serivce](https://github.com/hanxiao/bert-as-service，支持 Tensorflow
  - 默认的句向量构造方案是取倒数第二层的隐状态值在token上的均值，即选用的层数是倒数第2层，池化策略是REDUCE_MEAN
- 使用 [transformers](https://github.com/huggingface/transformers/) 库实现自定义的用法
  - http 并发请求简易版
  - 函数调用
  - 更适合中文任务
  - 例如 [sentence-transformers](https://github.com/UKPLab/sentence-transformers)



检索方法（语义召回）

- 基于关键字检索（倒排索引检索）
  - tf-idf，bm25

- 基于向量的检索（语义索引检索）
  - 参考[ Query 理解和语义召回在知乎搜索中的应用](https://www.6aiq.com/article/1577969687897)
  - 负采样，局部负采样 + 全局负采样
    - 对所有 query 进行聚类（bert embedding + knn）
    - 在每个聚类中进行局部负采样
    - 同时进行全局负采样
  - 句向量微调训练，[Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://www.sbert.net/docs/usage/semantic_search.html) 
    - bert average embeddings（优于 CLS ？）
    - 实现参考 [Semantic Textual Similarity](https://www.sbert.net/examples/training/sts/README.html)
  - 语义检索推理（[semantic search](https://www.sbert.net/docs/usage/semantic_search.html)）
    - 适用于小数据集（小于 10万条数据）
    - 可以计算query和语料库中所有条目之间的余弦相似度（`torch.nn.CosineSimilarity` ，可以自定义函数，提前计算中间变量）
    - 然后使用 `np.argpartition` （最大堆算法）仅对前k个条目进行排序，节约时间 

