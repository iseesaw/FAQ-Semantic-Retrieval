#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-08-29 02:03:16
# @Author  : Kaiyan Zhang (minekaiyan@gmail.com)
# @Link    : https://github.com/iseesaw
# @Version : 1.0.0
import os
import pprint
import argparse
import pandas as pd
from tokenizers.pre_tokenizers import PreTokenizer
from tqdm import tqdm
from typing import Union, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import BertPreTrainedModel, BertModel, PreTrainedTokenizer, BertTokenizer, Trainer, TrainingArguments

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
"""
基于 Transformers Trainer 的 SiameseNetwork
"""


class InputExample:
    """
    Structure for one input example with texts, the label and a unique id
    """
    def __init__(self,
                 guid: str = '',
                 texts: List[str] = None,
                 texts_tokenized: List[List[int]] = None,
                 label: Union[int, float] = None):
        """
        Creates one InputExample with the given texts, guid and label

        str.strip() is called on both texts.

        :param guid
            id for the example
        :param texts
            the texts for the example
        :param texts_tokenized
            Optional: Texts that are already tokenized. If texts_tokenized is passed, texts must not be passed.
        :param label
            the label for the example
        """
        self.guid = guid
        self.texts = [text.strip()
                      for text in texts] if texts is not None else texts
        self.texts_tokenized = texts_tokenized
        self.label = label

    def __str__(self):
        return "<InputExample> label: {}, texts: {}".format(
            str(self.label), "; ".join(self.texts))


class SiameseDataset(Dataset):
    def __init__(self, examples: List[InputExample],
                 tokenizer: PreTrainedTokenizer):
        """
        Create a new SentencesDataset with the tokenized texts and the labels as Tensor

        :param examples
            A list of sentence.transformers.readers.InputExample
        :param model
            SentenceTransformerModel
        """
        self.tokenizer = tokenizer
        self.examples = examples

    def __getitem__(self, item):
        """
        :return List[List[str]]
        """
        label = torch.tensor(self.examples[item].label, dtype=torch.long)

        # 分词, 使用 InputExample 保存避免多次重复
        if self.examples[item].texts_tokenized is None:
            self.examples[item].texts_tokenized = [
                self.tokenizer.tokenize(text)
                for text in self.examples[item].texts
            ]

        return self.examples[item].texts_tokenized, label

    def __len__(self):
        return len(self.examples)


class Collator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def batching_collate(self, batch):
        """
        使用 tokenizer 进行处理, 其中 is_pretokenized=True 
        在 Dataset 中提前进行分词(比较耗时)
        这里主要进行 token 转换生成模型需要的特征
        并对 sentence1/2 所在 batch 分别进行 padding

        :param batch:
            List[Tuple(Tuple(List[str], List[str]), int)]
            列表元素为 SiameseDataset.__getitem__() 返回结果
        :return:
            Dict[str, Union[torch.Tensor, Any]]
            包括 input_ids/attention_mask/token_type_ids 等特征
            为了区别 sentence1/2, 在特征名后面添加 '_1' 和 '_2'
        """
        # 分别获得 sentences 和 labels
        sentences, labels = map(list, zip(*batch))
        all_sentences = map(list, zip(*sentences))

        features = [
            self.tokenizer(sents,
                           return_tensors='pt',
                           padding=True,
                           truncation=True,
                           is_pretokenized=True,
                           max_length=self.max_length)
            for sents in all_sentences
        ]
        # 组合 sentence1/2 的特征, 作为 SiameseNetwork 的输入
        features1 = {k + '_1': v for k, v in features[0].items()}
        features2 = {k + '_2': v for k, v in features[1].items()}
        feat_dict = {**features1, **features2}

        labels = torch.tensor(labels, dtype=torch.long)
        feat_dict['labels'] = labels
        return feat_dict


class BertForSiameseNet(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)

        self.margin = args.margin

        self.bert = BertModel(config)

        self.init_weights()

    def _mean_pooling(self, model_output, attention_mask):
        """Mean Pooling - Take attention mask into account for correct averaging
        :param model_output: Tuple
        :param attention_mask: Tensor, (batch_size, seq_length)
        """
        # (batch_size, seq_length, hidden_size)
        token_embeddings = model_output[0]
        # First element of model_output contains all token embeddings
        # (batch_size, seq_length) => (batch_size, seq_length, hidden_size)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()).float()

        # Only sum the non-padding token embeddings
        # (batch_size, seq_length, hidden_size) => (batch_size, hidden_size)
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

        # smoothing, avoid being divided by zero
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self,
                input_ids_1=None,
                attention_mask_1=None,
                token_type_ids_1=None,
                input_ids_2=None,
                attention_mask_2=None,
                token_type_ids_2=None,
                labels=None):
        # sequence_output, pooled_output, (hidden_states), (attentions)
        outputs = [
            self.bert(input_ids_1,
                      attention_mask=attention_mask_1,
                      token_type_ids=token_type_ids_1),
            self.bert(input_ids_2,
                      attention_mask=attention_mask_2,
                      token_type_ids=token_type_ids_2),
        ]
        # 使用 mean pooling 获得句向量
        embeddings = [
            self._mean_pooling(output, mask) for output, mask in zip(
                outputs, [attention_mask_1, attention_mask_2])
        ]

        # 计算两个句向量的余弦相似度
        logits = F.cosine_similarity(embeddings[0], embeddings[1])
        outputs = (logits, )

        # 计算 onlineContrastiveLoss
        if labels is not None:
            distance_matrix = 1 - logits
            negs = distance_matrix[labels == 0]
            poss = distance_matrix[labels == 1]

            # select hard positive and hard negative pairs
            negative_pairs = negs[negs < (
                poss.max() if len(poss) > 1 else negs.mean())]
            positive_pairs = poss[poss > (
                negs.min() if len(negs) > 1 else poss.mean())]

            positive_loss = positive_pairs.pow(2).sum()
            negative_loss = F.relu(self.margin - negative_pairs).pow(2).sum()
            loss = positive_loss + negative_loss

            outputs = (loss, ) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


def compute_metrics(pred):
    def find_best_acc_and_threshold(scores, labels):
        """计算最高正确率的余弦距离阈值
        参考 https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/evaluation/BinaryClassificationEvaluator.py
        """
        # 按余弦距离从高到低排序
        rows = list(zip(scores, labels))
        rows = sorted(rows, key=lambda x: x[0], reverse=True)

        max_acc = 0
        best_threshold = -1

        positive_so_far = 0
        remaining_negatives = sum(labels == 0)

        # 遍历找到最高正确率的距离阈值
        for i in range(len(rows) - 1):
            _, label = rows[i]
            if label == 1:
                positive_so_far += 1
            else:
                remaining_negatives -= 1

            acc = (positive_so_far + remaining_negatives) / len(labels)
            if acc > max_acc:
                max_acc = acc
                best_threshold = (rows[i][0] + rows[i + 1][0]) / 2

        return max_acc, best_threshold

    labels = pred.label_ids
    scores = pred.predictions
    max_acc, best_threshold = find_best_acc_and_threshold(scores, labels)
    preds = [1 if p > best_threshold else 0 for p in scores]
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary')

    return {
        'threshold': best_threshold,
        'accuracy': max_acc,
        'f1': f1,
        'precision': precision,
        'recal': recall
    }


def load_sents_from_csv(filename):
    data = pd.read_csv(filename)
    examples = []
    for _, row in data.iterrows():
        examples.append(
            InputExample(texts=[row['sentence1'], row['sentence2']],
                         label=row['label']))

    print('loading %d examples from %s' % (len(examples), filename))
    return examples


def main(args):
    # 初始化预训练模型和分词器
    model = BertForSiameseNet.from_pretrained(
        args.model_name_or_path
        if args.mode == 'train' else args.model_save_path,
        args=args)
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)

    # 配置训练参数
    training_args = TrainingArguments(
        output_dir=args.model_save_path,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir=args.logging_dir)

    # 初始化 collator 用于批处理数据
    collator = Collator(tokenizer, args.max_length)

    if args.mode == 'train':
        # 读取训练集和开发集
        print('loading dataset')
        train_dataset = SiameseDataset(load_sents_from_csv(args.trainset_path),
                                       tokenizer)
        dev_dataset = SiameseDataset(load_sents_from_csv(args.devset_path),
                                     tokenizer)
        # 初始化训练器并开始训练
        print('start training')
        trainer = Trainer(model=model,
                          args=training_args,
                          data_collator=collator.batching_collate,
                          compute_metrics=compute_metrics,
                          train_dataset=train_dataset,
                          eval_dataset=dev_dataset)
        trainer.train()

        trainer.save_model()

    elif args.mode == 'test':
        test_samples = load_sents_from_csv(args.testset_path)
        test_dataset = SiameseDataset(test_samples, tokenizer)
        dev_dataset = SiameseDataset(load_sents_from_csv(args.devset_path),
                                     tokenizer)
        # 初始化训练器
        trainer = Trainer(model=model,
                          args=training_args,
                          data_collator=collator.batching_collate,
                          compute_metrics=compute_metrics)
        dev_result = trainer.evaluate(eval_dataset=dev_dataset)
        pprint.pprint(dev_result)

        threshold = dev_result['eval_threshold']

        labels = [sample.label for sample in test_samples]
        scores = trainer.predict(test_dataset=test_dataset).predictions

        # 计算正确率
        preds = [1 if s > threshold else 0 for s in scores]
        acc = accuracy_score(labels, preds)
        p_r_f1 = precision_recall_fscore_support(labels,
                                                 preds,
                                                 average='macro')
        print('accuracy', acc)
        print('macro_precision', p_r_f1[0])
        print('macro_recall', p_r_f1[1])
        print('macro_f1', p_r_f1[2])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Bert For Siamese Network')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--trainset_path',
                        type=str,
                        default='lcqmc/LCQMC_train.csv')
    parser.add_argument('--devset_path',
                        type=str,
                        default='lcqmc/LCQMC_dev.csv')
    parser.add_argument('--testset_path',
                        type=str,
                        default='lcqmc/LCQMC_test.csv')

    parser.add_argument(
        '--model_name_or_path',
        default='/users6/kyzhang/embeddings/bert/bert-base-chinese')
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--per_device_train_batch_size', type=int, default=64)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=128)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument(
        '--margin',
        type=float,
        default=0.5,
        help='Negative pairs should have a distance of at least 0.5')
    parser.add_argument('--model_save_path',
                        default='./output/transformers-bert-base-chinese')
    parser.add_argument('--logging_dir', type=str, default='./logs')

    args = parser.parse_args()
    main(args)