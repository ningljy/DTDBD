import os
import torch
import tqdm
import torch.nn as nn
import numpy as np
from .layers import *
from sklearn.metrics import *
from transformers import BertModel
from transformers import RobertaModel

class MLP(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)


class EDDFNModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, domain_num, dropout, dataset):
        super(EDDFNModel, self).__init__()
        if dataset == 'ch':
            self.bert = BertModel.from_pretrained('./pretrained_model/chinese-bert-wwm-ext').requires_grad_(False)
        elif dataset == 'en':
            self.bert = RobertaModel.from_pretrained('./pretrained_model/roberta-base').requires_grad_(False)

        self.shared_mlp = MLP(emb_dim, mlp_dims, dropout, False)
        self.specific_mlp = torch.nn.ModuleList([MLP(emb_dim, mlp_dims, dropout, False) for i in range(9)])
        self.decoder = MLP(mlp_dims[-1] * 2, (64, emb_dim), dropout, False)
        self.classifier = torch.nn.Linear(2 * mlp_dims[-1], 1)
        self.domain_classifier = nn.Sequential(MLP(mlp_dims[-1], mlp_dims, dropout, False), torch.nn.ReLU(),
                                               torch.nn.Linear(mlp_dims[-1], domain_num))
        self.attention = MaskAttention(emb_dim)

    def forward(self, alpha=1, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        category = kwargs['category']
        bert_feature = self.bert(inputs, attention_mask=masks).last_hidden_state
        bert_feature, _ = self.attention(bert_feature, masks)
        specific_feature = []
        for i in range(bert_feature.size(0)):
            specific_feature.append(self.specific_mlp[category[i]](bert_feature[i].view(1, -1)))
        specific_feature = torch.cat(specific_feature)
        shared_feature = self.shared_mlp(bert_feature)
        feature = torch.cat([shared_feature, specific_feature], 1)
        rec_feature = self.decoder(feature)
        output = self.classifier(feature)

        reverse = ReverseLayerF.apply
        domain_pred = self.domain_classifier(reverse(shared_feature, alpha))
        out.append(logits)
        out.append(output)
        out.append(shared_feature)
        return torch.sigmoid(output.squeeze(1)), rec_feature, bert_feature, domain_pred