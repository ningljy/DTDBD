import os
import torch
import tqdm
import torch.nn as nn
import datetime
import numpy as np
from models.layers import *
from models.mdfend import MultiDomainFENDModel as MDFENDModel
from models.student import StudentModel
from models.student_ad import StudentModel as StudentADModel
from models.bigru import BiGRUModel
from models.bert import BertFNModel
from models.m3fend import M3FENDModel
from utils.utils import data2gpu, Averager, metrics, Recorder

from torch.nn import functional as F

def euclidean_dist(shared_feature):
    trans=shared_feature.T
    dist_matrix=torch.cdist(trans,trans)
    dist_matrix=dist_matrix.T
    return dist_matrix

def distillation(student_scores,teacher_scores,temp):
    loss_soft=F.kl_div(F.log_softmax(student_scores/temp,dim=1),F.softmax(teacher_scores/temp,dim=1),reduction="batchmean")
    return loss_soft*temp*temp

class Trainer():
    def __init__(self,
                 modelname1,
                 modelname2,
                 emb_dim,
                 mlp_dims,
                 usemul,
                 logits_shape,
                 use_cuda,
                 dataset,
                 lr,
                 dropout,
                 category_dict,
                 weight_decay,
                 save_param_dir,
                 semantic_num,
                 emotion_num,
                 style_num,
                 lnn_dim,
                 early_stop,
                 epoches,
                 train_loader,
                 val_loader,
                 test_loader,
                 path1,
                 path2,
                 Momentum=0.99,
                 ):
        self.modelname1=modelname1
        self.modelname2=modelname2
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.path1=path1
        self.path2=path2
        self.early_stop = early_stop
        self.epoches = epoches
        self.category_dict = category_dict
        self.use_cuda = use_cuda
        self.usemul = usemul
        self.logits_shape = logits_shape

        self.emb_dim = emb_dim
        self.mlp_dims = mlp_dims
        self.dropout = dropout
        self.semantic_num = semantic_num
        self.emotion_num = emotion_num
        self.style_num = style_num
        self.lnn_dim = lnn_dim
        self.dataset = dataset
        self.Momentum=Momentum
        if os.path.exists(save_param_dir):
            self.save_param_dir = save_param_dir
        else:
            self.save_param_dir = save_param_dir
            os.makedirs(save_param_dir)

    def train(self):
        print('modelname',self.modelname1,self.modelname2)
        if self.modelname1 == 'mdfend':
            self.teacher0=MDFENDModel(self.emb_dim, self.mlp_dims, len(self.category_dict), self.dropout, self.dataset,logits_shape=self.logits_shape)
        elif self.modelname1 == 'm3fend':
            self.teacher0 = M3FENDModel(self.emb_dim, self.mlp_dims, self.dropout, self.semantic_num, self.emotion_num,
                                   self.style_num, self.lnn_dim, len(self.category_dict), dataset=self.dataset,logits_shape=self.logits_shape)
        self.teacher1 =StudentADModel(self.emb_dim, self.mlp_dims, len(self.category_dict), self.dropout, dataset=self.dataset, logits_shape=self.logits_shape)
        if self.modelname2 == 'textcnn-u':
            self.model = StudentModel(self.emb_dim, self.mlp_dims, len(self.category_dict), self.dropout,
                                  dataset=self.dataset, logits_shape=self.logits_shape)
        elif self.modelname2=='bigru-u':
            self.model = BiGRUModel(self.emb_dim, 1, self.mlp_dims, self.dropout, self.dataset)
        

        if self.use_cuda:
            self.model = self.model.cuda()
            self.teacher1 = self.teacher1.cuda()
            self.teacher0=self.teacher0.cuda()
        lossfun = torch.nn.BCELoss()
        loss_fn2=torch.nn.MSELoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        recorder = Recorder(self.early_stop)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.98)
        self.teacher0 = torch.load(self.path1)
        self.teacher1=torch.load(self.path2)
        f1_me=[0,0]
        fd_me=[0,0]
        m=self.Momentum
        a=0.4
        for epoch in range(self.epoches):
            if epoch>=5:
                tr_f1 =(f1_me[1]-f1_me[0])/(f1_me[0]+1e-5)
                tr_fd =(fd_me[1]-fd_me[0])/(fd_me[0]+1e-5)
                a= m*a - (1-m) * (tr_fd - tr_f1)/ (abs(tr_fd) + abs(tr_f1))
                if a> 0.3:
                    a=0.3
                if a <0.1:
                    a=0.1
            self.model.train()
            self.teacher0.eval()
            self.teacher1.eval()
            train_data_iter = tqdm.tqdm(self.train_loader)
            avg_loss = Averager()
            for step_n, batch in enumerate(train_data_iter):
                batch_data = data2gpu(batch, self.use_cuda)
                label = batch_data['label']
                category = batch_data['category']
                optimizer.zero_grad()
                out = self.model(**batch_data)
                with torch.no_grad():
                    teacher0out=self.teacher0(**batch_data)
                    teacher1out=self.teacher1(**batch_data,alpha=-1)
                loss1=distillation(out[0],teacher0out[0], 4)
                loss2 = distillation(euclidean_dist(out[2]), euclidean_dist(teacher1out[4]), 4)
                loss3=lossfun(out[1],label.float())
                loss = a*loss1 + (0.4-a)*loss2 + 0.6*loss3
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (scheduler is not None):
                    scheduler.step()
                avg_loss.add(loss.item())
            print('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))
            results = self.test(self.val_loader, 0)
            if epoch==0:
                f1_me[1]=results['f1']
                fd_me[1]=results['FNED']+results['FPED']
            else:
                f1_me[0] = f1_me[1]
                fd_me[0] = fd_me[1]
                f1_me[1] = results['f1']
                fd_me[1] = results['FNED'] + results['FPED']
            mark = recorder.add(results)
            if mark == 'save':
                torch.save(self.model.state_dict(),
                           os.path.join(self.save_param_dir, 'parameter' +'StudentKDfrom_'+
                                        self.modelname1+'_'+
                                        str(self.early_stop)+'_'+
                                        str(self.logits_shape)+'_'+
                                        str(self.usemul)+self.dataset+
                                        '.pkl'))
            elif mark == 'esc':
                break
            else:
                continue
        self.model.load_state_dict(torch.load(os.path.join(self.save_param_dir, 'parameter'+'StudentKDfrom_'+
                                        self.modelname1+'_'+
                                        str(self.early_stop)+'_'+
                                        str(self.logits_shape)+'_'+
                                        str(self.usemul)+self.dataset+
                                        '.pkl')))
        results = self.test(self.test_loader, 1)
        print(results)

        return results, os.path.join(self.save_param_dir, 'parameter' +'StudentKDfrom_'+
                                        self.modelname1+'_'+
                                        str(self.early_stop)+'_'+
                                        str(self.logits_shape)+'_'+
                                        str(self.usemul)+self.dataset+
                                        '.pkl')

    def test(self, dataloader, testorval):
        pred = []
        label = []
        category = []
        shared_feature = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(batch, self.use_cuda)
                batch_label = batch_data['label']
                batch_category = batch_data['category']
                out = self.model(**batch_data, alpha=-1)
                batch_label_pred = out[1]
                feature = out[2]
                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_label_pred.detach().cpu().numpy().tolist())
                category.extend(batch_category.detach().cpu().numpy().tolist())
                shared_feature.extend(feature.detach().cpu().numpy().tolist())

        result = metrics(label, pred, category, self.category_dict)
        if testorval == 1:
            torch.save(self.model,
                       'recodertestpkl/' + self.modelname1+self.modelname2 + self.dataset + '_CKD.pkl')

        return result

    def testteacher0(self,dataloader, testorval):
        pred = []
        label = []
        category = []
        shared_feature = []
        self.teacher0.eval()
        data_iter = tqdm.tqdm(dataloader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(batch, self.use_cuda)
                batch_label = batch_data['label']
                batch_category = batch_data['category']
                out = self.teacher0(**batch_data)
                batch_label_pred = out[1]
                feature = out[2]
                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_label_pred.detach().cpu().numpy().tolist())
                category.extend(batch_category.detach().cpu().numpy().tolist())
                shared_feature.extend(feature.detach().cpu().numpy().tolist())
        resultlog={}
        mainresultlog={}
        result = metrics(label, pred, category, self.category_dict)
        return result
    def testteacher1(self,dataloader, testorval):
        pred = []
        label = []
        category = []
        shared_feature = []
        self.teacher1.eval()
        data_iter = tqdm.tqdm(dataloader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(batch, self.use_cuda)
                batch_label = batch_data['label']
                batch_category = batch_data['category']
                out = self.teacher1(**batch_data, alpha=-1)
                batch_label_pred = out[1]
                feature = out[2]
                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_label_pred.detach().cpu().numpy().tolist())
                category.extend(batch_category.detach().cpu().numpy().tolist())
                shared_feature.extend(feature.detach().cpu().numpy().tolist())
        resultlog={}
        mainresultlog={}
        result = metrics(label, pred, category, self.category_dict)
        return result
