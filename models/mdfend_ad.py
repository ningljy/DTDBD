import numpy as np
import os
import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel
from transformers import RobertaModel
from .layers import *

class MultiDomainFENDModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, domain_num, dropout, dataset, logits_shape):
        super(MultiDomainFENDModel, self).__init__()
        self.logits_shape=logits_shape
        self.domain_num = domain_num
        self.gamma = 10
        self.num_expert = 5
        self.fea_size = 256
        if dataset == 'ch1':
            self.bert = BertModel.from_pretrained('./pretrained_model/chinese-bert-wwm-ext').requires_grad_(False)
        elif dataset == 'en':
            self.bert = RobertaModel.from_pretrained('./pretrained_model/roberta-base').requires_grad_(False)

        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        expert = []
        for i in range(self.num_expert):
            expert.append(cnn_extractor(feature_kernel, emb_dim))
        self.expert = nn.ModuleList(expert)

        self.gate = nn.Sequential(nn.Linear(emb_dim, mlp_dims[-1]),
                                  nn.ReLU(),
                                  nn.Linear(mlp_dims[-1], self.num_expert),
                                  nn.Softmax(dim=1))

        self.attention = MaskAttention(emb_dim)

        self.domain_embedder = nn.Embedding(num_embeddings=self.domain_num, embedding_dim=emb_dim)
        self.specific_extractor = SelfAttentionFeatureExtract(multi_head_num=1, input_size=emb_dim,
                                                              output_size=self.fea_size)

        self.classifier = nn.Sequential(MLP(320, mlp_dims, dropout, False),
                                        torch.nn.Linear(mlp_dims[-1], 2))
        self.classifier1 = torch.nn.Linear(2, 1)
        self.domain_classifier = nn.Sequential(MLP(320, mlp_dims, dropout, False), torch.nn.ReLU(),
                                               torch.nn.Linear(mlp_dims[-1], domain_num))

    def forward(self, alpha, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        category = kwargs['category']
        init_feature = self.bert(inputs, attention_mask=masks).last_hidden_state

        feature, _ = self.attention(init_feature, masks)
        idxs = torch.tensor([index for index in category]).view(-1, 1).cuda()
        domain_embedding = self.domain_embedder(idxs).squeeze(1)

        gate_value = self.gate(domain_embedding)

        shared_feature = 0
        for i in range(self.num_expert):
            tmp_feature = self.expert[i](init_feature)
            shared_feature += (tmp_feature * gate_value[:, i].unsqueeze(1))

        # print(shared_feature.size())  64*320
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








