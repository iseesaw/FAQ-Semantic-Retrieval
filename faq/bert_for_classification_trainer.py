#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-08-30 12:08:23
# @Author  : Kaiyan Zhang (minekaiyan@gmail.com)
# @Link    : https://github.com/iseesaw
# @Version : 1.0.0
import ast
from argparse import ArgumentParser

import pandas as pd

import torch
from transformers import BertForSequenceClassification, BertTokenizerFast, TrainingArguments, Trainer


class SimDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        """
        :param encodings
            Dict(str, List[List[int]])
        :labels
            List[int]
        """
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        """
        :return item
            dict-like object, Dict(str, tensor)
        """
        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def load_dataset(filename):
    """
    :param filename
    :return [texts1, texts2], labels
    """
    df = pd.read_csv(filename)
    # array -> list
    return [df['sentence1'].values.tolist(),
            df['sentence2'].values.tolist()], df['label'].values.tolist()


def main(args):
    # 初始化预训练模型和分词器
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name_or_path)
    model = BertForSequenceClassification.from_pretrained(
        args.model_name_or_path)

    # 加载 csv 格式数据集
    train_texts, train_labels = load_dataset(args.trainset_path)
    dev_texts, dev_labels = load_dataset(args.devset_path)
    test_texts, test_labels = load_dataset(args.testset_path)
    # 预处理获得模型输入特征
    train_encodings = tokenizer(text=train_texts[0],
                                text_pair=train_texts[1],
                                truncation=True,
                                padding=True,
                                max_length=args.max_length)
    dev_encodings = tokenizer(text=dev_texts[0],
                              text_pair=dev_texts[1],
                              truncation=True,
                              padding=True,
                              max_length=args.max_length)
    test_encodings = tokenizer(text=test_texts[0],
                               text_pair=test_texts[1],
                               truncation=True,
                               padding=True,
                               max_length=args.max_length)

    # 构建 SimDataset 作为模型输入
    train_dataset = SimDataset(train_encodings, train_labels)
    dev_dataset = SimDataset(dev_encodings, dev_labels)
    test_dataset = SimDataset(test_encodings, test_labels)

    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=args.do_train,
        do_eval=args.do_eval,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit)

    # 初始化训练器并开始训练
    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=dev_dataset)

    if args.do_train:
        trainer.train()

    if args.do_predict:
        metrics = trainer.evaluate(test_dataset)
        print(metrics)
    # 保存模型和分词器
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)


if __name__ == '__main__':
    parser = ArgumentParser('Bert For Classification')

    parser.add_argument('--do_train', type=ast.literal_eval, default=False)
    parser.add_argument('--do_eval', type=ast.literal_eval, default=True)
    parser.add_argument('--do_predict', type=ast.literal_eval, default=True)

    parser.add_argument(
        '--model_name_or_path',
        default='/users6/kyzhang/embeddings/bert/bert-base-chinese')

    parser.add_argument('--trainset_path', default='lcqmc/LCQMC_train.csv')
    parser.add_argument('--devset_path', default='lcqmc/LCQMC_dev.csv')
    parser.add_argument('--testset_path', default='lcqmc/LCQMC_test.csv')

    parser.add_argument('--output_dir',
                        default='output/transformers-bert-for-classification')
    parser.add_argument('--max_length',
                        type=int,
                        default=128,
                        help='max length of sentence1 & sentence2')
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--per_device_train_batch_size', type=int, default=64)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=64)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--logging_dir', type=str, default='./logs')
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--save_total_limit', type=int, default=3)

    args = parser.parse_args()
    main(args)
