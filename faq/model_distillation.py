#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-08-29 02:45:10
# @Author  : Kaiyan Zhang (minekaiyan@gmail.com)
# @Link    : https://github.com/iseesaw
# @Version : 1.0.0

import os
import ast
import logging
import argparse
import numpy as np
from typing import NamedTuple

import torch
from torch._C import device
from torch.utils.data import DataLoader

from textbrewer import GeneralDistiller
from textbrewer import TrainingConfig, DistillationConfig

from transformers import BertTokenizer, BertConfig
from transformers import AdamW

from functools import partial

from transformers_trainer import SiameseDataset, BertForSiameseNet, load_sents_from_csv, Collator, compute_metrics
from distills.matches import matches

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO)
logger = logging.getLogger(__name__)


def BertForQASimpleAdaptor(batch, model_outputs):
    dict_obj = {'logits': model_outputs[1], 'hidden': model_outputs[2]}
    return dict_obj


class EvalPrediction(NamedTuple):
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        predictions (:obj:`np.ndarray`): Predictions of the model.
        label_ids (:obj:`np.ndarray`): Targets to be matched.
    """

    predictions: np.ndarray
    label_ids: np.ndarray


def predict(model, dataloader, device, step):
    model.eval()
    preds = []
    label_ids = []
    for batch in dataloader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        intputs = {k: v for k, v in batch.items() if k != 'labels'}
        labels = batch['labels']
        with torch.no_grad():
            logits, _ = model(**intputs)
            preds.extend(logits.cpu().numpy())
            label_ids.extend(labels.cpu().numpy())

    model.train()
    eval_result = compute_metrics(
        EvalPrediction(predictions=np.array(preds),
                       label_ids=np.array(label_ids)))
    logger.info('Step {} eval results {}'.format(step, eval_result))


def main(args):
    ##### prepare
    os.makedirs(args.output_dir, exist_ok=True)
    forward_batch_size = int(args.train_batch_size /
                             args.gradient_accumulation_steps)
    args.forward_batch_size = forward_batch_size
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ##### load bert config & tokenizer
    bert_config_T = BertConfig.from_json_file(
        os.path.join(args.teacher_model, 'config.json'))
    tokenizer = BertTokenizer.from_pretrained(args.teacher_model)

    bert_config_S = BertConfig.from_json_file(args.student_config)

    ##### load data & init dataloader
    train_dataset = SiameseDataset(load_sents_from_csv(args.train_file),
                                   tokenizer)
    num_train_steps = int(
        len(train_dataset) / args.train_batch_size) * args.num_train_epochs

    collator = Collator(tokenizer, args.max_seq_length)
    dataloader = DataLoader(train_dataset,
                            collate_fn=collator.batching_collate,
                            batch_size=args.train_batch_size,
                            drop_last=True)

    eval_dataset = SiameseDataset(load_sents_from_csv(args.eval_file),
                                  tokenizer)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=collator.batching_collate,
                                 batch_size=args.eval_batch_size)

    ##### build model and load checkpoint
    model_T = BertForSiameseNet(bert_config_T, args)
    model_S = BertForSiameseNet(bert_config_S, args)

    # load teacher
    state_dict_T = torch.load(os.path.join(args.teacher_model,
                                           'pytorch_model.bin'),
                              map_location=args.device)
    model_T.load_state_dict(state_dict_T)
    model_T.eval()

    # load student
    state_dict_S = torch.load(
        os.path.join(args.teacher_model, 'pytorch_model.bin'))
    state_weight = {
        k[5:]: v
        for k, v in state_dict_S.items() if k.startswith('bert.')
    }
    missing_keys, _ = model_S.bert.load_state_dict(state_weight, strict=False)
    assert len(missing_keys) == 0

    model_T.to(args.device)
    model_S.to(args.device)

    ##### training
    optimizer = AdamW(model_S.parameters(), lr=args.learning_rate)

    logger.info("***** Running training *****")
    logger.info("  Num orig examples = %d", len(train_dataset))
    logger.info("  Forward batch size = %d", forward_batch_size)
    logger.info("  Num backward steps = %d", num_train_steps)

    ##### distillation
    train_config = TrainingConfig(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_dir=args.output_dir,
        output_dir=args.output_dir,
        device=args.device)
    # matches
    intermediate_matches = matches.get('L3_hidden_mse') + matches.get(
        'L3_hidden_smmd')
    distill_config = DistillationConfig(
        temperature=args.temperature,
        intermediate_matches=intermediate_matches)

    adaptor_T = partial(BertForQASimpleAdaptor)
    adaptor_S = partial(BertForQASimpleAdaptor)

    distiller = GeneralDistiller(train_config=train_config,
                                 distill_config=distill_config,
                                 model_T=model_T,
                                 model_S=model_S,
                                 adaptor_T=adaptor_T,
                                 adaptor_S=adaptor_S)
    callback_func = partial(predict,
                            dataloader=eval_dataloader,
                            device=args.device)

    with distiller:
        distiller.train(optimizer,
                        dataloader=dataloader,
                        num_epochs=args.num_train_epochs,
                        callback=callback_func)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Distillation For BERT')
    parser.add_argument('--teacher_model',
                        type=str,
                        default='./output/transformers-bert-base-chinese')
    parser.add_argument('--student_config',
                        type=str,
                        default='./distills/bert_config_L3.json')
    parser.add_argument(
        '--bert_model',
        type=str,
        default='/users6/kyzhang/embeddings/bert/bert-base-chinese')
    parser.add_argument('--output_dir',
                        type=str,
                        default='./distills/outputs/bert_L3')
    parser.add_argument('--train_file',
                        type=str,
                        default='lcqmc/LCQMC_train.csv')
    parser.add_argument('--eval_file', type=str, default='lcqmc/LCQMC_dev.csv')

    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--num_train_epochs', type=int, default=30)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--print_every', type=int, default=200)
    parser.add_argument('--weight', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=8)
    parser.add_argument('--output_hidden_states',
                        type=ast.literal_eval,
                        default=True)
    parser.add_argument(
        '--margin',
        type=float,
        default=0.5,
        help='Negative pairs should have a distance of at least 0.5')
    args = parser.parse_args()
    main(args)