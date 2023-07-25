import numpy as np
import torch
import tqdm
from sklearn import  metrics as  metr
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.nn import functional as F
import os
from sklearn.datasets import load_iris,load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

class Recorder(): ##记录

    def __init__(self, early_step):
        self.max = {'f1': 0,'FPED': 1}
        self.cur = {'f1': 0,'FNED': 1}
        self.maxindex = 0
        self.curindex = 0
        self.early_step = early_step

    def add(self, x):
        self.cur = x
        self.curindex += 1
        print("curent", self.cur)
        return self.judge()

    def judge(self):
        if self.cur['f1'] > self.max['f1']:
            # if (self.cur['f1'] - self.max['f1']>=0.002) or ((self.cur['FNED']+self.cur['FPED'])<(self.max['FNED']+self.max['FPED'])):
            self.max = self.cur
            self.maxindex = self.curindex
            self.showfinal()
            return 'save'
        self.showfinal()
        if self.curindex - self.maxindex >= self.early_step:
            return 'esc'
        else:
            return 'continue'

    def showfinal(self):
        print("Max", self.max)
def metrics(y_true, y_pred, category, category_dict):
    res_by_category = {}
    metrics_by_category = {}
    reverse_category_dict = {}
    for k, v in category_dict.items():
        reverse_category_dict[v] = k
        res_by_category[k] = {"y_true": [], "y_pred": []}

    for i, c in enumerate(category):
        c = reverse_category_dict[c]
        res_by_category[c]['y_true'].append(y_true[i])
        res_by_category[c]['y_pred'].append(y_pred[i])

    for c, res in res_by_category.items():
        try:
            metrics_by_category[c] = {
                'auc': metr.roc_auc_score(res['y_true'], res['y_pred']).round(4).tolist()
            }
        except Exception as e:
            metrics_by_category[c] = {
                'auc': 0
            }

    metrics_by_category['overallauc'] = metr.roc_auc_score(y_true, y_pred, average='macro').round(4),

    metrics_by_category['overallauc'] = list(metrics_by_category['overallauc'])[0]

    y_pred = np.around(np.array(y_pred)).astype(int)

    metrics_by_category['f1'] = metr.f1_score(y_true, y_pred, average='macro').round(4)

    allcm = metr.confusion_matrix(y_true, y_pred)
    print(allcm)

    tn, fp, fn, tp = allcm[0][0], allcm[0][1], allcm[1][0], allcm[1][1]
    metrics_by_category['overallFNR'] = (fn / (tp + fn)).round(4)
    metrics_by_category['overallFPR'] = (fp / (fp + tn)).round(4)
    # metrics_by_category['overallFNR'] = list(metrics_by_category['overallFNR'])[0]
    # metrics_by_category['overallFPR'] = list(metrics_by_category['overallFPR'])[0]

    for c, res in res_by_category.items():
        # try:
        metrics_by_category[c]['auc'] = metr.roc_auc_score(res['y_true'], res['y_pred'], average='macro').round(4),
        metrics_by_category[c]['auc'] = list(metrics_by_category[c]['auc'])[0]

        metrics_by_category[c]['f1'] = metr.f1_score(res['y_true'], np.around(np.array(res['y_pred'])).astype(int),
                                                     average='macro').round(4).tolist(),
        metrics_by_category[c]['f1'] = list(metrics_by_category[c]['f1'])[0]

        cm = metr.confusion_matrix(res['y_true'], np.around(np.array(res['y_pred'])).astype(int))
        tn1, fp1, fn1, tp1 = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

        metrics_by_category[c]['FNR'] = (fn1 / (tp1 + fn1)).round(4)
        metrics_by_category[c]['FPR'] = (fp1 / (fp1 + tn1)).round(4)

        # confusion_matrix = metr.confusion_matrix(res['y_true'], np.around(np.array(res['y_pred'])).astype(int)),
        # confusion_matrix= list(confusion_matrix)1
        # metrics_by_category[c]['FNR']=confusion_matrix[0][1][0]/(confusion_matrix[0][1][0]+confusion_matrix[0][1][1]).round(4).tolist(),
        # metrics_by_category[c]['FPR']=(confusion_matrix[0][0][1]/(confusion_matrix[0][0][0]+confusion_matrix[0][0][1])).round(4).tolist(),
        # # metrics_by_category[c]['FNR'] = list(metrics_by_category[c]['FNR'])[0]
        # # metrics_by_category[c]['FPR'] = list(metrics_by_
        # category[c]['FNR'])[0]

        # except Exception as e:
        #     metrics_by_category[c]['auc']=0,
        #     metrics_by_category[c]['FNR']=0,
        #     metrics_by_category[c]['FPR']=0,

    metrics_by_category['FNED'] = 0
    metrics_by_category['FPED'] = 0

    for k, v in category_dict.items():
        metrics_by_category['FNED'] += abs(metrics_by_category['overallFNR'] - metrics_by_category[k]['FNR'])
        metrics_by_category['FPED'] += abs(metrics_by_category['overallFPR'] - metrics_by_category[k]['FPR'])
        metrics_by_category['FPED'] = metrics_by_category['FPED'].round(4)
        metrics_by_category['FNED'] = metrics_by_category['FNED'].round(4)

    return metrics_by_category


def data2gpu(batch, use_cuda):
    if use_cuda:
        batch_data = {
            'content': batch[0].cuda(),
            'content_masks': batch[1].cuda(),
            'comments': batch[2].cuda(),
            'comments_masks': batch[3].cuda(),
            'content_emotion': batch[4].cuda(),
            'comments_emotion': batch[5].cuda(),
            'emotion_gap': batch[6].cuda(),
            'style_feature': batch[7].cuda(),
            'label': batch[8].cuda(),
            'category': batch[9].cuda(),
            'categoryonehot': batch[10].cuda()
            }
    else:
        batch_data = {
            'content': batch[0],
            'content_masks': batch[1],
            'comments': batch[2],
            'comments_masks': batch[3],
            'content_emotion': batch[4],
            'comments_emotion': batch[5],
            'emotion_gap': batch[6],
            'style_feature': batch[7],
            'label': batch[8],
            'category': batch[9],
            'categoryonehot': batch[10]
            }
    return batch_data

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v