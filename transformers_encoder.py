#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-08-29 01:51:01
# @Author  : Kaiyan Zhang (minekaiyan@gmail.com)
# @Link    : https://github.com/iseesaw
# @Version : 1.0.0
import torch
from transformers import AutoTokenizer, AutoModel


class TransformersEncoder:
    """封装基于Transformers的句向量编码器
    """
    def __init__(
        self,
        model_name_or_path='/users6/kyzhang/embeddings/bert/bert-base-chinese',
        max_length=128):
        """初始化

        Args:
            model_name_or_path (str, optional): Transformers模型位置或者别称（从HuggingFace服务器下载）
            max_length (int, optional): 最大长度. Defaults to 128.
        """
        print('initing encoder')
        print('loading model from from pretrained')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)

        # gpu & cpu
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('using', self.device)
        self.model.to(self.device) 
        self.model.eval()
        
        self.max_length = max_length
        print('ending initing')


    def _assign_device(self, Tokenizer_output):
        """将tensor转移到gpu
        """
        tokens_tensor = Tokenizer_output['input_ids'].to(self.device)
        token_type_ids = Tokenizer_output['token_type_ids'].to(self.device)
        attention_mask = Tokenizer_output['attention_mask'].to(self.device)

        output = {'input_ids' : tokens_tensor, 
                'token_type_ids' : token_type_ids, 
                'attention_mask' : attention_mask}

        return output

    def _mean_pooling(self, model_output, attention_mask):
        """平均池化

        Args:
            model_output ([type]): transformers 模型输出
            attention_mask (List[List[int]]): MASK, (batch, seq_length)

        Returns:
            List[List[int]]: 句向量
        """
        # (batch_size, seq_length, hidden_size)
        token_embeddings = model_output[0].cpu()

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
        """句向量编码器

        Args:
            sentences (List[str]): (batch_size)

        Returns:
            Tensor: (batch_size, hidden_size)
        """
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

if __name__ == '__main__':
    # pip install transformers==3.0.2 ([Optional] torch==1.6.0)
    # https://github.com/huggingface/transformers
    encoder = TransformersEncoder()
    # (batch_size, hidden_size)
    print(encoder.encode(['你好呀']))