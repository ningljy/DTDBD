import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--max_len', type=int, default=170)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--Momentum', type=int, default=0.99)
parser.add_argument('--gpu', default='0')
parser.add_argument('--emb_dim', type=int, default=768)
parser.add_argument('--model_name', default='m3fend')
parser.add_argument('--model_name2', default='student')
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--usemul', type=int,default=0)
parser.add_argument('--early_stop', type=int, default=5)
parser.add_argument('--dataset', default='ch1')# en
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--domain_num', type=int, default=9)
parser.add_argument('--logits_shape', type=int, default=2)
parser.add_argument('--save_log_dir', default= './logs')
parser.add_argument('--save_param_dir', default= './param_model')
parser.add_argument('--param_log_dir', default = './logs/param')
parser.add_argument('--semantic_num', type=int, default=7)
parser.add_argument('--emotion_num', type=int, default=7)
parser.add_argument('--style_num', type=int, default=2)
parser.add_argument('--lnn_dim', type=int, default=50)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
import numpy as np
import torch
import random
from Combined_KD_m import Trainer as CKDTrainer
from utils.dataloader import bert_data
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
config = {
        'model_name2':args.model_name2,
        'use_cuda': True,
        'usemul':args.usemul,
        'batchsize': args.batchsize,
        'max_len': args.max_len,
        'early_stop': args.early_stop,
        'num_workers': args.num_workers,
        'logits_shape':args.logits_shape,
        'dataset': args.dataset,
        'root_path': './data/ch1/',
        'weight_decay': 5e-5,
        'category_dict': {
                    "科技": 0,
                    "军事": 1,
                    "教育考试": 2,
                    "灾难事故": 3,
                    "政治": 4,
                    "医药健康": 5,
                    "财经商业": 6,
                    "文体娱乐": 7,
                    "社会生活": 8,
                },
        'mlp_dims':[384],
        'dropout':0.2,
        'emb_dim': args.emb_dim,
        'lr': args.lr,
        'epoch': args.epoch,
        'model_name': args.model_name,
        'seed': args.seed,
        'semantic_num': args.semantic_num,
        'emotion_num': args.emotion_num,
        'style_num': args.style_num,
        'domain_num': args.domain_num,
        'lnn_dim': args.lnn_dim,#the number of cross-view representations
        'save_log_dir': args.save_log_dir,
        'save_param_dir': args.save_param_dir,
        'param_log_dir': args.param_log_dir,
        'Momentum': args.Momentum,
        }


def get_dataloader(train_path,val_path,test_path,category_dict,dataset):
    loader = bert_data(max_len=config['max_len'], batch_size=config['batchsize'],
                       category_dict=category_dict, num_workers=config['num_workers'], dataset=dataset,
                       domain_num=config['domain_num'])
    train_loader = loader.load_data(train_path, True)
    val_loader = loader.load_data(val_path, False)
    test_loader = loader.load_data(test_path, False)
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    if config['dataset'] == 'en':
        config['domain_num']=3
        config['root_path'] = './data/en/'
        config['category_dict'] = {
            "gossipcop": 0,
            "politifact": 1,
            "COVID": 2,
        }
    elif config['dataset'] == 'ch1':
        config['root_path'] = './data/ch1/'
        if args.domain_num == 9:
            config['category_dict'] = {
                "科技": 0,
                "军事": 1,
                "教育考试": 2,
                "灾难事故": 3,
                "政治": 4,
                "医药健康": 5,
                "财经商业": 6,
                "文体娱乐": 7,
                "社会生活": 8,
            }
    train_loader, val_loader, test_loader = get_dataloader(config['root_path'] + 'train.pkl',
                                                           config['root_path'] + 'val.pkl',
                                                           config['root_path'] + 'test.pkl',
                                                           config['category_dict'],
                                                           config['dataset'])
    config['model_name1'] =config['model_name']
    config['model_name2'] =config['model_name2']
    path1 = './recodertestpkl/' + config['model_name1'] + config['dataset'] + '.pkl'
    path2 = './recodertestpkl/' + config['model_name2'] + config['dataset'] + '_ad.pkl'
    trainer = CKDTrainer(config['model_name1'], config['model_name2'], config['emb_dim'], config['mlp_dims'],
                         config['usemul'], 2,
                         config['use_cuda'], config['dataset'], config['lr'], config['dropout'],
                         config['category_dict'], config['weight_decay'],
                         config['save_param_dir'], config['semantic_num'], config['emotion_num'], config['style_num'],
                         config['lnn_dim'],
                         config['early_stop'], config['epoch'], train_loader, val_loader, test_loader, path1, path2,config['Momentum'],)

    trainer.train()


