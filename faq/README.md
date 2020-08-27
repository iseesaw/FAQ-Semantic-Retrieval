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
