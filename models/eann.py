import os
import torch
import tqdm
import torch.nn as nn
import numpy as np
from .layers import *
from transformers import BertModel
from transformers import RobertaModel

class EANNModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, domain_num, dropout, dataset,logits_shape):
        super(EANNModel, self).__init__()
        self.logits_shape = logits_shape
        if dataset == 'ch1':
            self.bert = BertModel.from_pretrained('./pretrained_model/chinese-bert-wwm-ext').requires_grad_(False)
        elif dataset == 'en':
            self.bert = RobertaModel.from_pretrained('./pretrained_model/roberta-base').requires_grad_(False)

        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        self.convs = cnn_extractor(feature_kernel, emb_dim)
        mlp_input_shape = sum([feature_kernel[kernel] for kernel in feature_kernel])


        self.classifier = nn.Sequential(MLP(320, mlp_dims, dropout, False),
                                        torch.nn.Linear(mlp_dims[-1], 2))
        self.classifier1 = torch.nn.Linear(2, 1)

        self.domain_classifier = nn.Sequential(MLP(mlp_input_shape, mlp_dims, dropout, False), torch.nn.ReLU(),
                        torch.nn.Linear(mlp_dims[-1], domain_num))
    
    def forward(self, alpha, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']

        bert_feature = self.bert(inputs, attention_mask = masks).last_hidden_state

        shared_feature = self.convs(bert_feature)
        out = []
        reverse = ReverseLayerF.apply
        domain_pred = self.domain_classifier(reverse(shared_feature, alpha))
        domain_pred1 = self.domain_classifier(reverse(shared_feature, -alpha))
        logits = self.classifier(shared_feature)
        output = self.classifier1(logits)
        output = torch.sigmoid(output.squeeze(1))
        out.append(logits)
        out.append(output)
        out.append(domain_pred)
        out.append(domain_pred1)
        out.append(shared_feature)
        return out


