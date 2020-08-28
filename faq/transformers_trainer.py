#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-08-29 02:03:16
# @Author  : Kaiyan Zhang (minekaiyan@gmail.com)
# @Link    : https://github.com/iseesaw
# @Version : 1.0.0
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel, AutoTokenizer, Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
'''
基于 Trainer 的版本
可以自定义模块，参考 BertForSequenceClasification
https://github.com/huggingface/transformers/blob/9336086ab5d232cccd9512333518cf4299528882/src/transformers/modeling_bert.py#L1277
'''


class BertForFAQ(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits, ) + outputs[
            2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels),
                                labels.view(-1))
            outputs = (loss, ) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmx(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recal': recall}


def train():
    '''
    trainer 使用参考 https://colab.research.google.com/drive/1-JIJlao4dI-Ilww_NnTc0rxtp-ymgDgM?usp=sharing
    '''
    model = BertForFAQ.from_pretrained('bert-base-chinese')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')

    train_dataset, test_dataset = [], []

    training_args = TrainingArguments(output_dir='./results',
                                      num_train_epochs=3,
                                      per_device_train_batch_size=16,
                                      per_device_eval_batch_size=64,
                                      warmup_steps=500,
                                      weight_decay=0.01,
                                      logging_dir='./logs')

    trainer = Trainer(model=model,
                      args=training_args,
                      compute_metrics=compute_metrics,
                      train_dataset=train_dataset,
                      eval_dataset=test_dataset)

    trainer.train()


if __name__ == '__main__':
    train()