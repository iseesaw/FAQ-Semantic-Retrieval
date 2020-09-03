#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-08-26 01:27:12
# @Author  : Kaiyan Zhang (minekaiyan@gmail.com)
# @Link    : https://github.com/iseesaw
# @Version : 1.0.0

from sentence_transformers import models, SentenceTransformer


# Model
def get_model(model_name_or_path, device='cuda'):
    """初始化 SentenceTransformer 编码器

    Args:
        model_name_or_path (str): Transformers 或者微调后的 BERT 模型
        device (str, optional): cpu or cuda. Defaults to 'cuda'.

    Returns:
        SentenceTransformers: 编码器
    """
    word_embedding_model = models.BERT(model_name_or_path)

    # 使用 mean pooling 获得句向量表示
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model],
                                device=device)

    return model


if __name__ == '__main__':
    # pip install sentence-transformers
    # https://github.com/UKPLab/sentence-transformers
    model = get_model('./output/transformers-merge3-bert-6L')
    # （hidden_size）
    print(model.encode('你好呀'))
    # （batch_size, hidden_size）
    # print(model.encode(['你好呀']))