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


```python
from transformers import AutoTokenizer, AutoModel
import torch


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask



#Sentences we want sentence embeddings for
sentences = ['This framework generates embeddings for each input sentence',
             'Sentences are passed as a list of string.',
             'The quick brown fox jumps over the lazy dog.']

#Load AutoModel from huggingface model repository
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
model = AutoModel.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")

#Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')

#Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

#Perform pooling. In this case, mean pooling
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
```