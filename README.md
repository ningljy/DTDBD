# Dual-Teacher De-biasing Distillation Framework for Multi-domain Fake News Detection(DTDBD)

This is a concrete implementation of our paper"Dual-Teacher De-biasing Distillation Framework for Multi-domain Fake News Detection"presented at ICDE2024.

Multi-domain fake news detection aims to identify whether various news from different domains is real or fake and has become urgent and important. However, existing methods are dedicated to improving the overall performance of fake news detection, ignoring the fact that unbalanced data leads to disparate treatment for different domains, i.e., the domain bias problem. To solve this problem, we propose the Dual-Teacher De-biasing Distillation framework (DTDBD) to mitigate bias across different domains. Following the knowledge distillation methods, DTDBD adopts a teacher-student structure, where pre-trained large teachers instruct a student model. In particular, the DTDBD consists of an unbiased teacher and a clean teacher that jointly guide the student model in mitigating domain bias and maintaining performance. For the unbiased teacher, we introduce an adversarial de-biasing distillation loss to instruct the student model in learning unbiased domain knowledge. For the clean teacher, we design domain knowledge distillation loss, which effectively incentivizes the student model to focus on representing domain features while maintaining performance. Moreover, we present a momentum-based dynamic adjustment algorithm to trade off the effects of two teachers. Extensive experiments on Chinese and English datasets show that the proposed method substantially outperforms the state-of-the-art baseline methods in terms of bias metrics while guaranteeing competitive performance.

## Introduction

This repository provides specific implementations of the DTDBD framework and 13 baselines, please note that our student models are TextCNN-U and BiGRU-U.

## Requirements

Python 3.8

PyTorch>1.0

Pandas

Numpy

Tqdm

Transformers

## Run

dataset: the English(en) or Chinese dataset(ch1)

gpu: the index of gpu you will use, default for `0`

model_name: model_name within textcnn bigru bert eann eddfn mmoe mose dualemotion stylelstm mdfend m3fend.

You can run this code through to train baseline model:

```
python main.py --gpu 0 --lr 0.0001 --model_name m3fend --dataset ch1
```

You can run this code to get the baseline model that joins adversarial learning :

```
python mainad.py --gpu 0 --lr 0.0001 --model_name textcnn-u --dataset ch1
```

You can run this code to run the DTDBD framework:

```
python mainCKD.py --gpu 0 --lr 0.0001 --teacher_name m3fend --student_name textcnn-u --dataset ch1
```
## Reference
