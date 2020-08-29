# FAQ之基于BERT的语义召回

## 介绍

当用户输入 query 后，FAQ 的处理流程一般为

- **问题理解**，对 query 进行改写以及向量表示

- **召回模块**，在大规模问题集中进行候选问题召回，获得 topk

- **排序模块**，对召回后续问题集合进行排序，然后返回概率最大后续问题对应的答案/回复



本文主要讨论 **召回模块**，即获取 topk 后直接返回，主要内容包括：

- **FAQ 召回方法**，包括**基于关键字的倒排索引**（使用 ElasticSearch 或 Lucene 或 Whoosh）；基于**向量检索的语义召回**，使用 BERT 预训练模型表示问题，并使用余弦相似度计算距离（使用 sentence-transformers 或 transformers 实现）

- **BERT 微调与蒸馏**，构造训练数据集，在 BERT 模型基础上 fine-tune，进行模型蒸馏

- **FAQ 线上服务部署**，使用进程管理以及缓存等机制，进行压力测试



## FAQ 召回方法

> 基于关键字倒排索引和基于向量的语义召回实现思路

### 关键字检索（倒排索引）

计算候选问题集的 TF-IDF 或者 BM25 得分，构建倒排索引表，可以使用 Python 第三方库进行索引和查询 [ElasticSearch](https://whoosh.readthedocs.io/en/latest/index.html)，[Lucene](https://lucene.apache.org/pylucene/)，[Whoosh](https://whoosh.readthedocs.io/en/latest/index.html)



### 向量检索（语义召回）

离线使用 BERT 等预训练模型对候选集合中所有问题进行编码得到 corpus 向量矩阵，当用户输入 query 时，同样进行编码得到向量表示，计算 query 向量和 corpus 向量矩阵的余弦相似度，找回 topk 候选问题

向量表示可以使用如下几种方法

|                                                              | 优点                             | 缺点                               |
| ------------------------------------------------------------ | -------------------------------- | ---------------------------------- |
| [bert-as-serivce](https://github.com/hanxiao/bert-as-service) | 高并发                           | 自定义程度低<br />难以扩展其他模型 |
| [Sentence-Transformers](https://www.sbert.net/index.html)    | 支持多种模型                     |                                    |
| [transformers](https://github.com/huggingface/transformers/) | 支持多种模型<br />自定义程度最高 |                                    |



#### Bert-as-service

 [bert-as-serivce](https://github.com/hanxiao/bert-as-service) 是基于 Google 官方代码实现的 BERT 服务，可以本地或者HTTP请求进行句子编码

其默认的句向量构造方案是取倒数第二层的隐状态值在token上的均值，即选用的层数是倒数第2层，池化策略是 REDUCE_MEAN

```python
# 需要运行服务 bert-serving-start -model_dir /tmp/english_L-12_H-768_A-12/ -num_worker=4 
from bert_serving.client import BertClient

# 创建客户端示例并进行编码
bc = BertClient()
bc.encode(['First do it', 'then do it right', 'then do it better'])
```

通过 HTTP 请求

```python
# 运行 http 服务 bert-serving-start -model_dir=/YOUR_MODEL -http_port 8125
import requests
reply = requests.post('http://127.0.0.1:8125/encode',
                     data={
                        "id": 123,
                        "texts": ["hello world", "good day!"],
                        "is_tokenized": false
                    })
reply.json()
# {
#    "id": 123,
#    "results": [[768 float-list], [768 float-list]],
#    "status": 200
# }
```



#### Sentence-transformers

> 尽量使用最新版本，否则可能报错
>
> ```bash
> torch                              1.6.0
> sentence-transformers              0.3.4
> transformers                       3.0.2
> ```

[Sentence-Transformers](https://www.sbert.net/index.html) 是一个基于 [Transformers](https://huggingface.co/transformers/) 库的句向量表示 Python 框架，内置语义检索以及多种文本（对）相似度损失函数，可以较快的实现模型和语义检索，并提供多种[预训练的模型](https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/)，包括中文在内的多语言模型

```python
from sentence_transformers import SentenceTransformer, model

# 加载内置模型 (https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/)
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

sentences = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.',
    'The quick brown fox jumps over the lazy dog.']

# 句子编码
embeddings = model.encode(sentences)
```

> 如果需要加载 HuggingFace 的预训练模型，需要指定特定的类，比如使用 `bert-base-chinese` ，[#issue75](https://github.com/UKPLab/sentence-transformers/issues/75#issuecomment-568717443)
>
> ```python
> from sentence_transformers import models
> 
> model_name = 'bert-base-chinese'
> 
> # 使用 BERT 作为 encoder
> word_embedding_model = models.BERT(model_name)
> 
> # 使用 mean pooling 获得句向量表示
> pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
>                                pooling_mode_mean_tokens=True,
>                                pooling_mode_cls_token=False,
>                                pooling_mode_max_tokens=False)
> 
> model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
> ```



#### Transformers

[Transformers](https://huggingface.co/transformers/) 支持多种预训练模型，可扩展性强，预训练模型默认输出 `sequence_output` 和 `pool_output` 分别为句子中每个 token 的向量表示和 [CLS] 的向量表示（相关论文证明 Average Embedding 效果好于 [CLS]）

```python
from transformers import AutoTokenizer, AutoModel
import torch


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    # 获得句子中每个 token 的向量表示
    token_embeddings = model_output[0] 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


sentences = ['This framework generates embeddings for each input sentence',
             'Sentences are passed as a list of string.',
             'The quick brown fox jumps over the lazy dog.']

# 加载模型文件
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# 数据预处理
encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')

# 模型推理
with torch.no_grad():
    model_output = model(**encoded_input)

# mean pooling 获得句子表示
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
```



## BERT 微调与蒸馏

### 数据集

[文本相似度数据集](https://github.com/IceFlameWorm/NLP_Datasets)

FAQ 数据集

内部 FAQ 数据集

### 负采样

对于每个 query，需要获得与其相似的 **positve candidate** 以及不相似的 **negtive candidate**，从而构成正样本和负样本作为模型输入，即 **(query, candidate)**

⚠️ 此处为 **offline negtive sampling**，即在训练前采样构造负样本，区别于 **online negtive sampling**，后者在训练中的每个 batch 内进行动态的负采样（可以通过相关损失函数实现）

两种方法可以根据任务特性进行选择，**online negtive sampling** 对于数据集有一定的要求，需要确保每个 batch 内的 query 是不相似的，但是效率更高

对于 **offline negtive sampling** 主要使用以下两种方法：

- **全局负采样**

  在整个数据集上进行正态分布采样，很难产生高难度的负样本

- **局部负采样**

  首先使用少量人工标注数据预训练的 **BERT 模型**对候选问题集合进行**编码**，然后使用**无监督聚类** ，如 [Kmeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)，最后在每个 query 所在聚类簇中采样负样本

以上所有数据综合作为最终训练数据

> 可以使用 **LCQMC** 标注的数据进行 BERT 模型预训练，但是 **LCQMC** 数据主要为百度知道问题，与FAQ中日常对话存在一定差异的
>
> [LCQMC: A Large-scale Chinese Question Matching Corpus](https://www.semanticscholar.org/paper/LCQMC%3A-A-Large-scale-Chinese-Question-Matching-Liu-Chen/549c1a581b61f9ea47afc6f6871845392eaebbc4)



### 双塔模型

语义召回模块使用**基于交互的双塔模型** （也称为 **Siamese Net** ，孪生网络）

- **Encoding**，使用 **BERT** 分别对 query 和 candidate 进行编码
- **Pooling**，将最后一层的**平均词嵌入作为句子表示**（或者 [CLS] 的向量表示，效果存在差别）
- **Computing**，计算两个向量的**点积**（或 **余弦相似度**）作为最终得分

$$
BERT_{rep}(q, d) = dot(\frac{1}{L} \sum_{t=1}^{L}{\vec q_i^{last}}, \frac{1}{L} \sum_{i=1}^{L} {\vec c_i^{last}})
$$

- **双塔模型**如下（其中 $cosine-sim(u,v)$ 也可以是其他的**度量函数**）

<img src="https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/SBERT_Siamese_Network.png" alt="SBERT Siamese Network Architecture" style="zoom: 50%;" />

> **基于表示的模型**，将 query 和 cand 拼接作为 BERT 的输入，然后取最后一层 [CLS] 字段的向量表示输入全连接层进行分类
> $$
> BERT_{rel}(q, c) = \vec w \times \vec {qd}_{cls}^{last}
> $$
> **基于表示的模型**比基于交互的模型更好，但是速度较慢， 一般用于排序模块
>
> 此外，也可以使用 [poly-encoder](https://arxiv.org/abs/1905.01969) 在基于表示和交互之间进行折中考虑



### 损失函数

> 参考博客 [Understanding Ranking Loss, Contrastive Loss, Margin Loss, Triplet Loss, Hinge Loss and all those confusing names](https://gombru.github.io/2019/04/03/ranking_loss/) 
>
> 同时结合  [SentenceTransformers](https://www.sbert.net/index.html) 中 [Loss API](https://www.sbert.net/docs/package_reference/losses.html) 总结

排序模型使用的损失函数为 **Ranking loss**，不同于C rossEntropy 和 MSE 进行分类和回归任务，**Ranking loss** 目的是预测输入样本对（即上述双塔模型中 $u$ 和 $v$ 之间）之间的相对距离（**度量学习任务**）



- **Contrastive Loss**

  > 来自 LeCun [Dimensionality Reduction by Learning an Invariant Mapping](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf)
  >
  > [sentence-transformers 源码](https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/ContrastiveLoss.py)

  - 公式形式如下，其中 $u, v$ 为 BERT 编码的向量表示，$y$ 为对应的标签（1 表示正样本，0 表示负样本）， $\tau$ 为超参数
    $$
    L(u_i, v_i, y_i) =  y_i ||u_i, v_i|| + (1 - y_i) \max(0, \tau - ||u_i, v_i||
    $$

  - 公式意义为：对于**正样本**，输出特征向量之间距离要**尽量小**；而对于**负样本**，输出特征向量间距离要**尽量大**；但是若**负样本间距太大**（即容易区分的**简单负样本**，间距大于 $\tau$）**则不处理**，让模型关注更加**难以区分**的样本



- **OnlineContrastive Loss**

  - 属于 **online negative sampling** ，与 **Contrastive Loss** 类似

  - 参考 [sentence-transformers 源码](https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/OnlineContrastiveLoss.py) ，在每个 batch 内，选择最难以区分的正例样本和负例样本进行 loss 计算（容易识别的正例和负例样本则忽略）

  - 公式形式如下，如果正样本距离小于 $\tau_1$ 则不处理，如果负样本距离大于 $\tau_0$ 则不处理，实现过程中 $\tau_0, \tau_1$ 可以分别取负/正样本的平均距离值
    $$
    L(u_i, v_i, y_i) 
    \begin{cases}
    \max (0, ||u_i, v_i|| - \tau_1) ,\ if \ y_i=1 & \\
    \max (0, \tau_0 - ||u_i, v_i||),\ if \ y_i =0
    \end{cases}
    $$



- **Multiple Negatives Ranking Loss**

  > 来自Google [Efficient Natural Language Response Suggestion for Smart Reply](https://arxiv.org/pdf/1705.00652.pdf)
  >
  > [sentence-transformers 源码](https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/MultipleNegativesRankingLoss.py)

  - 属于 **online negative sampling** ，当如果只有正样本（例如只有相似的段落/问答对等）时，训练过程中在每个 batch 内进行负采样
  - 输入句子对 $(a_1, b_1), (a_2, b_), \cdots, (a_n, b_n)$ ，假设 $(a_i, b_i)$ 是正样本，$(a_i, b_j), i \ne j$  是负样本
  - 在每个 batch 内，对于 $a_i$ ，其使用所有其他的 $b_j$ 作为负样本，然后优化 negative log-likehood loss
  - 依赖于负样本之间的距离，如果距离差距小，则效果可能不好



- **Cosine Similarity Loss**

  - 对于每个样本的 query 和 candidate 之间计算余弦相似度，然后与标准相似度之间计算 MSE 损失

  - 对于输入的标注相似度（即 label）需要是 float 类型（如果标签为 0 或 1，拟合效果可能欠佳）

  - 公式形式如下
    $$
    L(u, v, y) = (\frac{u \cdot v}{||u||_2 \times ||v||_2} - y)^2
    $$



- **Pairwise Max Margin Hinge loss**

  > 知乎搜索系统论文中用到的损失函数
  >
  > [Beyond Lexical: A Semantic Retrieval Framework for Textual SearchEngine](http://arxiv.org/abs/2008.03917)

  - 运用 SVM 中 max margin 思想，尽量拉远正负样本之间的间距
  - 训练过程中在每个 batch 内，度量两个正负样本（pairwise）之间的距离
  - 公式如下，其中 $p_i$ 和 $p_j$ 分别表示每个 $<query, candidate>$ 的模型计算得分，$y_i$ 和 $y_j$ 表示每个候选问题的标签，$\tau$ 为用于对齐的超参数

  $$
  \frac{1}{M} \sum_i \sum_{j \ne i} \max (0, \tau - (y_i - y_j) \times (p_i -p_j))
  $$



### fine-tune

fine-tune 可以使用高度封装的 [Sentence-Transformers](https://www.sbert.net/index.html)  快速搭建模型，或者使用 [Transformers](https://huggingface.co/transformers/) 更灵活地实现自定义模型

| 框架                                                      | 优点                                                         | 缺点               |
| --------------------------------------------------------- | ------------------------------------------------------------ | ------------------ |
| [Sentence-Transformers](https://www.sbert.net/index.html) | 简单易用<br />内置多种 Ranking loss<br />插件式模块使用<br />主要用于句对分类任务 | 不支持多GPU训练    |
| [Transformers](https://huggingface.co/transformers/)      | 内置多种训练机制<br />包括半精度、分布式训练等<br />自定义程度较高，适合各种任务 | Ranking loss自定义 |



- [Sentence-Transformers](https://www.sbert.net/index.html) 

  - 可以进行多种任务的fine-tune，这里可以选择类似的 [Quora Duplicate Questions](https://www.sbert.net/examples/training/quora_duplicate_questions/README.html) 作为参考，即判断两个问题是否相似，标签为 0 和 1

  - 首先需要初始化预训练模型，`model_name` 必须是 Sentece-Transformers [内置的预训练模型](https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/)

    ```python
    from sentence_transformers import SentenceTransformer, models
    # 加载内置预训练模型
    model = SentenceTransformer('model_name')
    
    # 可以自定义模块 或 使用 HugginFace Transformers 预训练模型
    #word_embedding_model = models.Transformer('bert-base-chinese', max_seq_length=256)
    # 可以选择使用 cls/max/mean 等多种 pooling 方式
    #pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    #model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    ```

  - 之后创建训练集，通过内置的 InputExample 构建样本

    ```python
    from torch.utils.data import DataLoader
    from sentence_transformers import SentencesDataset
    from sentence_transformers.readers import InputExample
    
    # 每个样本使用 InputExample 进行封装
    train_samples = [InputExample(texts=['sentence1', 'sentence2'], label=1)]
    train_dataset = SentencesDataset(train_samples, model=model)
    train_dataloader = DataLoader(train_dataset,
                                  shuffle=True,
                                  batch_size=64)
    # 使用余弦距离作为度量指标 (cosine_distance = 1-cosine_similarity)
    distance_metric = losses.SiameseDistanceMetric.COSINE_DISTANCE
    # 初始化训练损失函数
    train_loss = losses.OnlineContrastiveLoss(model=model,
                                              distance_metric=distance_metric,
                                              margin=0.5)
    ```

  - 然后可以针对具体任务评估器

    - 简单的分类，给定 (sentence1, sentence2) 通过计算两个句向量的余弦相似度判断是否相似，如果高于某个阈值则判断为相似（源码对开发集结果进行排序，依次取两个相邻得分结果作为阈值，遍历查找最高的 f1 值作为结果）

    ```python
    from sentence_transformers import evaluation
    from sentence_transformer import losses
    
    dev_sents1, dev_sents2, dev_labels = ['sent1', 'sent2'], ['sent11', 'sent22'], [1, 0]
    binary_acc_evaluator = evaluation.BinaryClassificationEvaluator(
            dev_sentences1, dev_sentences2, dev_labels)
    ```

    - 还可以构建多种评估器，使用 `evaluation.SequentialEvaluator()` 依次进行评估，参考 [training_OnlineConstrativeLoss.py ](https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/quora_duplicate_questions/training_OnlineConstrativeLoss.py) 同时进行分类以及检索等评估

  - 最后进行模型训练

    ```python
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=binary_acc_evaluator,
              epochs=10,
              warmup_steps=1000,
              output_path='output',
              output_path_ignore_not_empty=True)
    ```

> 高度封装，训练流程和 sklearn 以及 keras 类似，简单容易理解
>
> 不支持多GPU训练，[Multi-GPU-training #311](https://github.com/UKPLab/sentence-transformers/issues/311#issuecomment-659455875)
>
> 更多例子参考 [sentence-transformers/examples](https://github.com/UKPLab/sentence-transformers/tree/master/examples)



- [Transformers](https://huggingface.co/transformers/)

  - 除了使用 [SentenceTransformers](https://www.sbert.net/index.html) ，也可以直接使用 [Transformers](https://huggingface.co/transformers/) 的 [Trainer](https://huggingface.co/transformers/main_classes/trainer.html) 接口进行快速模型训练
- 读取所有句对保存 List[InputExample]
  - 初始化 SiameseDataset(examples, tokenizer)
- 实现 `__getitem__` 方法，返回句对的 id 表示，Tuple(List[int], List[int]), label
  
  - 实现 DataLoader 的 collator 方法
    - 重点是在每个batch内部进行补齐，但需要速度快，那么分词得提前完成
    - 输入 List[Tuple(List[int], List[int]), label]，输出 {features: [{input: tensor, mask: tensor}, {intput: tensor, mask:tensor}], labels: Tensor()}
    - features[0]，features[1] 分别进行 padding
    - tocuda
  
  - 调用模型，分别输入 features，{input: tensor, mask: tensor}，获得 batch 的句向量
  - 然后计算句对的余弦相似度并返回 loss



### 模型蒸馏

 [TextBrewer](https://github.com/airaria/TextBrewer) 基于 Transformers 的模型蒸馏工具



## FAQ Web服务搭建

### Flask 搭建WebAPI

- Flask + Gunicorn + gevent + nginx ，进程管理（自启动）（uwsgi同理，前者更简单）
-  [werkzeug.contrib.cache](https://werkzeug.palletsprojects.com/en/0.16.x/contrib/cache/)  使用 cache 缓存， 注意版本 werkzeug==0.16.0



### Locust 压力测试





## 更多

- 本文适用于小规模数据（小于10w？），对于更大规模的检索可以参考 [facebookresearch/faiss](3https://github.com/facebookresearch/faiss) （建立向量索引，然后使用 k-nearest- neighbor 进行召回）

- 很多场景下，基于关键字的倒排索引召回结果已经足够，可以考虑综合基于关键字和基于向量的召回方法，参考 [Query 理解和语义召回在知乎搜索中的应用](https://www.6aiq.com/article/1577969687897)



