import numpy as np
import os
import torch
from .layers import *
from torch import nn
import torch.nn.functional as F
from transformers import BertModel
from transformers import RobertaModel

class StudentModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims,domain_num, dropout, dataset,logits_shape):
        super(StudentModel, self).__init__()
        if dataset == 'ch1':
            self.bert = BertModel.from_pretrained('./pretrained_model/chinese-bert-wwm-ext').requires_grad_(False)
            # self.bert = BertModel.from_pretrained('hfl/chinese-bert-wwm-ext').requires_grad_(False)

        elif dataset == 'en':
            self.bert = RobertaModel.from_pretrained('./pretrained_model/roberta-base').requires_grad_(False)
            # self.bert = RobertaModel.from_pretrained('roberta-base',cache_dir='./pretrained_model').requires_grad_(False)


        for name,param in self.bert.named_parameters():
            if name.startswith("encoder.layer.11"):
                param.requires_grad = True
        # self.em=self.bert.embeddings
        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64}
        self.convs1 = cnn_extractor(feature_kernel, emb_dim)
        # self.convs2 = cnn_extractor(feature_kernel, emb_dim)

        mlp_input_shape = sum([feature_kernel[kernel] for kernel in feature_kernel])
        # self.mutiheadattention=nn.MultiheadAttention(768,8,batch_first=True)
        # self.LSTM=nn.LSTM(input_size=emb_dim,hidden_size=768,num_layers=1,batch_first=True,bidirectional=False).requires_grad_(True)
        # self.attention=MaskAttention(emb_dim)
        # self.sharefeaturecl=torch.nn.Linear(1408,320)
        self.classifier = nn.Sequential(MLP(mlp_input_shape, mlp_dims, dropout, False),
                                        torch.nn.Linear(mlp_dims[-1], 2))
        self.classifier1 = torch.nn.Linear(2, 1)
    def forward(self, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        bert_feature = self.bert(inputs, attention_mask=masks).last_hidden_state
        shared_feature=self.convs1(bert_feature)
        out = []
        logits = self.classifier(shared_feature)
        output = self.classifier1(logits)
        output = torch.sigmoid(output.squeeze(1))
        out.append(logits)         #0
        out.append(output)         #1
        out.append(shared_feature) #2
        return out