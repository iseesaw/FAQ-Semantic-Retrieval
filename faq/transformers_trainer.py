#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-08-29 02:03:16
# @Author  : Kaiyan Zhang (minekaiyan@gmail.com)
# @Link    : https://github.com/iseesaw
# @Version : 1.0.0
import os
import argparse
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import BertPreTrainedModel, BertModel, BertTokenizer, Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
'''
基于 Trainer 的版本
可以自定义模块，参考 BertForSequenceClasification
https://github.com/huggingface/transformers/blob/9336086ab5d232cccd9512333518cf4299528882/src/transformers/modeling_bert.py#L1277
'''


class BertForSiameseNet(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)

        self.margin = args.margin

        self.bert = BertModel(config)

        self.init_weights()

    def _mean_pooling(self, model_output, attention_mask):
        '''Mean Pooling - Take attention mask into account for correct averaging
        :param model_output: Tuple
        :param attention_mask: Tensor, (batch_size, seq_length)
        '''
        # (batch_size, seq_length, hidden_size)
        token_embeddings = model_output[0].cpu()

        # First element of model_output contains all token embeddings
        # (batch_size, seq_length) => (batch_size, seq_length, hidden_size)
        input_mask_expanded = attention_mask.cpu().unsqueeze(-1).expand(
            token_embeddings.size()).float()

        # Only sum the non-padding token embeddings
        # (batch_size, seq_length, hidden_size) => (batch_size, hidden_size)
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

        # smoothing, avoid being divided by zero
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                labels=None):
        # sequence_output, pooled_output, (hidden_states), (attentions)
        outputs = [
            self.bert(ids, attention_mask=masks,
                      token_type_ids=types) for ids, masks, types in zip(
                          input_ids, attention_mask, token_type_ids)
        ]

        embeddings = [
            self._mean_pooling(output, mask)
            for output, mask in zip(outputs, attention_mask)
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


class SiameseDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length):
        sentences1, sentences2, labels = dataset
        features1 = tokenizer(sentences1,
                              return_tensors='pt',
                              max_length=max_length,
                              padding=True,
                              truncation=True)
        features2 = tokenizer(sentences2,
                              return_tensors='pt',
                              max_length=max_length,
                              padding=True,
                              truncation=True)
        self.examples = [
            (feat1, feat2, label)
            for feat1, feat2, label in zip(features1, features2, labels)
        ]

    def __getitem__(self, item):
        example = self.examples[item]
        features = {
            k: torch.stack([example[0][k], example[1][k]])
            for k in example[0]
        }
        features['label'] = torch.tensor(self.examples[item][2],
                                         dtype=torch.long)
        return features

    def __len__(self):
        return len(self.examples)


def compute_metrics(pred):
    def find_best_acc_and_threshold(scores, labels):
        '''计算最高正确率的余弦距离阈值
        Ref https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/evaluation/BinaryClassificationEvaluator.py
        '''
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
    sents1, sents2, labels = [], [], []
    for _, row in data.iterrows():
        sents1.append(row['sentence1'])
        sents2.append(row['sentence2'])
        labels.append(row['label'])

    return (sents1, sents2, labels)


def main(args):
    # 初始化预训练模型和分词器
    model = BertForSiameseNet.from_pretrained(args.model_name_or_path,
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

    if args.mode == 'train':
        # 读取训练集和开发集
        print('loading dataset')
        train_dataset = SiameseDataset(load_sents_from_csv(args.trainset_path),
                                       tokenizer,
                                       max_length=args.max_length)
        dev_dataset = SiameseDataset(load_sents_from_csv(args.devset_path),
                                     tokenizer,
                                     max_length=args.max_length)
        # 初始化训练器并开始训练
        print('start training')
        trainer = Trainer(model=model,
                          args=training_args,
                          compute_metrics=compute_metrics,
                          train_dataset=train_dataset,
                          eval_dataset=dev_dataset)
        trainer.train()

        trainer.save_model()

    elif args.mode == 'test':
        test_samples = load_sents_from_csv(args.testset_path)
        test_dataset = SiameseDataset(test_samples,
                                      tokenizer,
                                      max_length=args.max_length)
        dev_dataset = SiameseDataset(load_sents_from_csv(args.devset_path),
                                     tokenizer,
                                     max_length=args.max_length)
        # 初始化训练器并开始训练
        trainer = Trainer(model=model,
                          args=training_args,
                          compute_metrics=compute_metrics)
        dev_result = trainer.evaluate(eval_dataset=dev_dataset)
        threshold = dev_result['threshold']

        labels = test_samples[2]
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