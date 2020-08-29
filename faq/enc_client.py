#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-08-27 11:46:29
# @Author  : Kaiyan Zhang (minekaiyan@gmail.com)
# @Link    : https://github.com/iseesaw
# @Version : 1.0.0

import torch
from transformers import AutoTokenizer, AutoModel


class EncodeClient:
    '''封装基于 Transformers 的句向量编码器
    '''
    def __init__(
        self,
        model_name_or_path='/users6/kyzhang/embeddings/bert/bert-base-chinese',
        max_length=128):
        '''
        :param model_name_or_path: str, pre-trained model name of path,
                                which is for loading model from huggingface model repository
        :param max_length: int, max length of sentences
        '''
        print('initing encode client')
        # Load AutoModel from huggingface model repository
        print('loading model from from pretrained')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)

        # Using GPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('using', self.device)
        self.model.to(self.device)

        self.max_length = max_length
        print('ending initing')


    def _assign_device(self, Tokenizer_output):

        tokens_tensor = Tokenizer_output['input_ids'].to(self.device)
        token_type_ids = Tokenizer_output['token_type_ids'].to(self.device)
        attention_mask = Tokenizer_output['attention_mask'].to(self.device)

        output = {'input_ids' : tokens_tensor, 
                'token_type_ids' : token_type_ids, 
                'attention_mask' : attention_mask}

        return output

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


    def encode(self, sentences):
        '''
        :param sentences: List[str], (batch_size)
        :return: Tensor (batch_size, hidden_size)
        '''
        # Tokenize sentences
        encoded_input = self.tokenizer(sentences,
                                       padding=True,
                                       truncation=True,
                                       max_length=self.max_length,
                                       return_tensors='pt')
        
        encoded_input = self._assign_device(encoded_input)
        # Compute token embeddings
        with torch.no_grad():
            # (sequence_output, pooled_output, (hidden_states), (attentions))
            # sequence_output, (batch_size, sequence_length, hidden_size))
            #   Sequence of hidden-states at the output of the last layer of the model.
            # pooled_output, (batch_size, hidden_size))
            #   Last layer hidden-state of the first token of the sequence (classification token) 
            #   further processed by a Linear layer and a Tanh activation function. 
            #   The Linear layer weights are trained from the next sentence prediction 
            #   (classification) objective during pre-training.
            #   not a good summary of the semantic content of the input
            #   it's better with averaging or pooling the sequence of hidden-states for the whole input sequence
            model_output = self.model(**encoded_input)

        # Perform pooling. In this case, mean pooling
        sentence_embeddings = self._mean_pooling(
            model_output, encoded_input['attention_mask'])

        return sentence_embeddings