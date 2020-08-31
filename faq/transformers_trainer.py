#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-08-29 02:03:16
# @Author  : Kaiyan Zhang (minekaiyan@gmail.com)
# @Link    : https://github.com/iseesaw
# @Version : 1.0.0
import ast
import logging
import pprint
import argparse
import pandas as pd
from typing import Union, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import BertConfig
from transformers import BertPreTrainedModel, BertModel, PreTrainedTokenizer, BertTokenizer
from transformers import Trainer, TrainingArguments

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
"""
基于 BERT 的 SiameseNetwork, 并使用 Transformers Trainer 进行模型训练
"""
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample:
    def __init__(self,
                 guid: str = '',
                 texts: List[str] = None,
                 texts_tokenized: List[List[int]] = None,
                 label: Union[int, float] = None):
        """
        :param guid
            样本唯一标识ID
        :param texts
            句对文本
        :param texts_tokenized
            [可选] 是否已经分词
        :param label
            样本标签
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
        构建 siamese 数据集, 主要为分词后的句对以及标签

        :param examples
            List[InputExample]
        :param tokenizer
            PreTrainedTokenizer
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
        self.output_hidden_states = args.output_hidden_states

        self.bert = BertModel(config)

        self.init_weights()

    def _mean_pooling(self, model_output, attention_mask):
        """Mean Pooling - Take attention mask into account for correct averaging
        :param model_output: Tuple
        :param attention_mask: Tensor, (batch_size, seq_length)
        """
        # embedding_output + attention_hidden_states x 12
        #hidden_states = model_output[2]
        # (batch_size, seq_length, hidden_size)
        token_embeddings = model_output[0]  #hidden_states[3]

        # (batch_size, seq_length) => (batch_size, seq_length, hidden_size)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()).float()

        # 只将原始未补齐部分的词嵌入进行相加
        # (batch_size, seq_length, hidden_size) => (batch_size, hidden_size)
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

        # 1e-9 用于平滑, 避免除零操作
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
        output1 = self.bert(input_ids_1,
                            attention_mask=attention_mask_1,
                            token_type_ids=token_type_ids_1,
                            output_hidden_states=self.output_hidden_states)
        output2 = self.bert(input_ids_2,
                            attention_mask=attention_mask_2,
                            token_type_ids=token_type_ids_2)
        outputs = [output1, output2]
        # 使用 mean pooling 获得句向量
        embeddings = [
            self._mean_pooling(output, mask) for output, mask in zip(
                outputs, [attention_mask_1, attention_mask_2])
        ]

        # 计算两个句向量的余弦相似度
        logits = F.cosine_similarity(embeddings[0], embeddings[1])
        outputs = (
            logits,
            output1[2],
        ) if self.output_hidden_states else (logits, )

        # 计算 onlineContrastiveLoss
        if labels is not None:
            distance_matrix = 1 - logits
            negs = distance_matrix[labels == 0]
            poss = distance_matrix[labels == 1]

            # 选择最难识别的正负样本
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

    # 真实和预测标签
    labels = pred.label_ids
    scores = pred.predictions

    # 计算最佳正确率时的距离阈值
    max_acc, best_threshold = find_best_acc_and_threshold(scores, labels)

    # 计算此时的精确率/召回率和F1值
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
    """加载数据集并保存为 InputExample
    """
    data = pd.read_csv(filename)
    examples = []
    for _, row in data.iterrows():
        examples.append(
            InputExample(texts=[row['sentence1'], row['sentence2']],
                         label=row['label']))

    logger.info('Loading %d examples from %s' % (len(examples), filename))
    return examples


def main(args):
    # 初始化预训练模型和分词器
    model_path = args.model_name_or_path if args.do_train else args.output_dir
    # import os
    # config = BertConfig.from_json_file('./distills/bert_config_L3.json')
    # model = BertForSiameseNet(config, args=args)
    # checkpoint = torch.load(os.path.join(model_path, 'pytorch_model.bin'))
    # model.load_state_dict(checkpoint, strict=False)
    model = BertForSiameseNet.from_pretrained(model_path, args=args)
    tokenizer = BertTokenizer.from_pretrained(model_path)

    # 配置训练参数, 更多配置参考文档
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments
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
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit)

    # 初始化 collator 用于批处理数据
    collator = Collator(tokenizer, args.max_length)

    if args.do_train:
        logger.info('*** TRAIN ***')
        # 读取训练集和开发集
        logger.info('Loading train/dev dataset')
        train_dataset = SiameseDataset(load_sents_from_csv(args.trainset_path),
                                       tokenizer)
        dev_dataset = SiameseDataset(load_sents_from_csv(args.devset_path),
                                     tokenizer)
        # 初始化训练器并开始训练
        logger.info('Start training')
        trainer = Trainer(model=model,
                          args=training_args,
                          data_collator=collator.batching_collate,
                          compute_metrics=compute_metrics,
                          train_dataset=train_dataset,
                          eval_dataset=dev_dataset)
        trainer.train()

        # 保存模型和词表
        logger.info('Save model and tokenizer to %s' % args.output_dir)
        trainer.save_model()
        tokenizer.save_pretrained(args.output_dir)

    elif args.do_predict:
        logger.info('*** TEST **')
        # 加载开发集和测试集
        test_samples = load_sents_from_csv(args.testset_path)
        test_dataset = SiameseDataset(test_samples, tokenizer)
        dev_dataset = SiameseDataset(load_sents_from_csv(args.devset_path),
                                     tokenizer)
        # 初始化训练器
        trainer = Trainer(model=model,
                          args=training_args,
                          data_collator=collator.batching_collate,
                          compute_metrics=compute_metrics)

        # 获得开发集结果（计算阈值）
        dev_result = trainer.evaluate(eval_dataset=dev_dataset)
        threshold = dev_result['eval_threshold']

        # 计算测试集结果及正确率（使用开发集阈值）
        labels = [sample.label for sample in test_samples]
        scores = trainer.predict(test_dataset=test_dataset).predictions

        preds = [1 if s > threshold else 0 for s in scores]
        acc = accuracy_score(labels, preds)
        p_r_f1 = precision_recall_fscore_support(labels,
                                                 preds,
                                                 average='macro')
        test_result = {
            'accuracy': acc,
            'macro_precision': p_r_f1[0],
            'macro_recall': p_r_f1[1],
            'macro_f1': p_r_f1[2]
        }
        logger.info('Test results {}'.format(test_result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Bert For Siamese Network')
    parser.add_argument('--do_train', type=ast.literal_eval, default=False)
    parser.add_argument('--do_eval', type=ast.literal_eval, default=True)
    parser.add_argument('--do_predict', type=ast.literal_eval, default=True)
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
    parser.add_argument('--per_device_train_batch_size', type=int, default=256)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=256)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--save_steps', type=int, default=1000)
    parser.add_argument('--save_total_limit', type=int, default=3)
    parser.add_argument(
        '--margin',
        type=float,
        default=0.5,
        help='Negative pairs should have a distance of at least 0.5')
    parser.add_argument('--output_hidden_states',
                        type=ast.literal_eval,
                        default=False)
    parser.add_argument('--output_dir',
                        default='./output/transformers-bert-L3')
    parser.add_argument('--logging_dir', type=str, default='./logs')

    args = parser.parse_args()
    main(args)