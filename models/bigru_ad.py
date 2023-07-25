import os
import torch
import tqdm
import torch.nn as nn
import numpy as np
from .layers import *
from transformers import BertModel
from transformers import RobertaModel
class BiGRUModel(torch.nn.Module):
    def __init__(self, emb_dim,num_layers, mlp_dims,domain_num, dropout, dataset):
        super(BiGRUModel, self).__init__()
        self.fea_size = emb_dim

        if dataset == 'ch1':
            self.bert = BertModel.from_pretrained('./pretrained_model/chinese-bert-wwm-ext').requires_grad_(False)
        elif dataset == 'en':
            self.bert = RobertaModel.from_pretrained('./pretrained_model/roberta-base').requires_grad_(False)

        self.rnn = nn.GRU(input_size=emb_dim,
                          hidden_size=self.fea_size,
                          num_layers=num_layers,
                          batch_first=True,
                          bidirectional=True)

        input_shape = self.fea_size * 2
        self.attention = MaskAttention(input_shape)
        self.classifier = nn.Sequential(MLP(input_shape, mlp_dims, dropout, False),
                                        torch.nn.Linear(mlp_dims[-1], 2))
        self.classifier1 = torch.nn.Linear(2, 1)
        self.domain_classifier = nn.Sequential(MLP(input_shape, mlp_dims, dropout, False), torch.nn.ReLU(),
                                               torch.nn.Linear(mlp_dims[-1], domain_num))
    def forward(self,alpha, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        bert_feature = self.bert(inputs, attention_mask=masks).last_hidden_state
        feature, _ = self.rnn(bert_feature)
        shared_feature, _ = self.attention(feature, masks)
        out = []
        reverse = ReverseLayerF.apply
        domain_pred = self.domain_classifier(reverse(shared_feature, alpha))
        domain_pred1 = self.domain_classifier(shared_feature)
        logits = self.classifier(shared_feature)
        output = self.classifier1(logits)
        output = torch.sigmoid(output.squeeze(1))
        out.append(logits)
        out.append(output)
        out.append(domain_pred)
        out.append(domain_pred1)
        out.append(shared_feature)
        return out

