import os
import torch
import tqdm
import torch.nn as nn
import numpy as np
from .layers import *
from transformers import BertModel
from transformers import RobertaModel

class BiGRUModel(torch.nn.Module):
    def __init__(self, emb_dim, num_layers, mlp_dims, dropout, dataset):
        super(BiGRUModel, self).__init__()
        self.fea_size = emb_dim
        if dataset == 'ch1':
            self.bert = BertModel.from_pretrained('./pretrained_model/chinese-bert-wwm-ext').requires_grad_(False)
        elif dataset == 'en':
            self.bert = RobertaModel.from_pretrained('./pretrained_model/roberta-base').requires_grad_(False)
        self.rnn = nn.GRU(input_size = emb_dim,
                          hidden_size = self.fea_size,
                          num_layers = num_layers, 
                          batch_first = True, 
                          bidirectional = True)

        input_shape = self.fea_size * 2
        self.attention = MaskAttention(input_shape)
        self.classifier = nn.Sequential(MLP(input_shape, mlp_dims, dropout, False),
                                        torch.nn.Linear(mlp_dims[-1], 2))

        self.classifier1 = torch.nn.Linear(2, 1)
    def forward(self, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        bert_feature = self.bert(inputs, attention_mask=masks).last_hidden_state
        feature, _ = self.rnn(bert_feature)
        shared_feature, _ = self.attention(feature, masks)
        out = []
        logits = self.classifier(shared_feature)
        output = self.classifier1(logits)
        output = torch.sigmoid(output.squeeze(1))
        out.append(logits)
        out.append(output)
        out.append(shared_feature)
        return out

