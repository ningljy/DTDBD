import os
import torch
import torch.nn as nn
import numpy as np
from .layers import *
from transformers import BertModel
from transformers import RobertaModel

class BertFNModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims,domain_num, dropout, dataset):
        super(BertFNModel, self).__init__()
        if dataset == 'ch1':
            self.bert = BertModel.from_pretrained('./pretrained_model/chinese-bert-wwm-ext').requires_grad_(False)
        elif dataset == 'en':
            self.bert = RobertaModel.from_pretrained('./pretrained_model/roberta-base').requires_grad_(False)
        self.attention = MaskAttention(emb_dim)
        self.classifier = nn.Sequential(MLP(emb_dim, mlp_dims, dropout, False),
                                        torch.nn.Linear(mlp_dims[-1], 2))
        self.classifier1 = torch.nn.Linear(2, 1)
    def forward(self, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        bert_feature = self.bert(inputs, attention_mask=masks).last_hidden_state
        shared_feature, _ = self.attention(bert_feature, masks)
        out = []
        logits = self.classifier(shared_feature)
        output = self.classifier1(logits)
        output = torch.sigmoid(output.squeeze(1))
        out.append(logits)
        out.append(output)
        out.append(shared_feature)
        return out
