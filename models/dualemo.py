import os
import torch
import tqdm
import torch.nn as nn
import numpy as np
from .layers import *
from sklearn.metrics import *
from transformers import BertModel
from transformers import RobertaModel

class DualEmotionModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, dropout, dataset):
        super(DualEmotionModel, self).__init__()
        self.fea_size = emb_dim

        if dataset == 'ch1':
            self.bert = BertModel.from_pretrained('./pretrained_model/chinese-bert-wwm-ext').requires_grad_(False)
        elif dataset == 'en':
            self.bert = RobertaModel.from_pretrained('./pretrained_model/roberta-base').requires_grad_(False)
        self.rnn = nn.GRU(input_size=emb_dim,
                          hidden_size=self.fea_size,
                          num_layers=1,
                          batch_first=True,
                          bidirectional=True)
        self.attention = MaskAttention(self.fea_size * 2)
        if dataset == 'ch1':
            self.classifier = nn.Sequential(MLP(self.fea_size * 2 + 47 * 5, mlp_dims, dropout, False),
                                            torch.nn.Linear(mlp_dims[-1], 2))
        elif dataset == 'en':
            self.classifier = nn.Sequential(MLP(self.fea_size * 2 + 38 * 5, mlp_dims, dropout, False),
                                            torch.nn.Linear(mlp_dims[-1], 2))
        self.classifier1 = torch.nn.Linear(2, 1)
    def forward(self, **kwargs):
        content = kwargs['content']
        content_masks = kwargs['content_masks']
        content_emotion = kwargs['content_emotion']
        comments_emotion = kwargs['comments_emotion']
        emotion_gap = kwargs['emotion_gap']
        emotion_feature = torch.cat([content_emotion, comments_emotion, emotion_gap], dim=1)

        content_feature = self.bert(content, attention_mask=content_masks)[0]
        content_feature, _ = self.rnn(content_feature)
        content_feature, _ = self.attention(content_feature, content_masks)

        shared_feature = torch.cat([content_feature, emotion_feature], dim=1)
        out = []
        logits = self.classifier(shared_feature)
        output = self.classifier1(logits)
        output = torch.sigmoid(output.squeeze(1))
        out.append(logits)
        out.append(output)
        out.append(shared_feature)
        return out
