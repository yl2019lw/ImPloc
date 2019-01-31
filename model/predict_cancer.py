#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cancerloader
import os
import torch
import numpy as np
import pandas as pd
from util import torch_util
from util import datautil

META_DIR = '/ndata/longwei/hpa/cancer_3tissue_list'
# NORMAL_META_DIR = '/ndata/longwei/hpa/tissuedata'
NORMAL_META_DIR = '/ndata/longwei/hpa/all_tissuedata'


CANCER_FV_DIR = '/ndata/longwei/hpa/cancerfv_all'
NORMAL_FV_DIR = '/ndata/longwei/hpa/normalfv_all'
NUM_CLASSES = 6

tissue_list = ['liver cancer', 'breast cancer', 'prostate cancer']
normal_tissue_list = ['liver', 'breast', 'prostate']


def predict():
    test_data = cancerloader.load_cancer_data()
    torch.nn.Module.dump_patches = True

    model_name = "ubploss/transformer_res18-128_size0_autoloss_alpha16"
    model_dir = os.path.join("./modeldir/%s" % model_name)
    model_pth = os.path.join(model_dir, "model.pth")

    model = torch.load(model_pth).cuda()

    model.eval()
    with torch.no_grad():
        all_gene = []
        all_gt = []
        all_pd = []
        for item in cancerloader.batch_fv(test_data, len(test_data)):
            genes, nimgs, labels, timesteps = item

            inputs = torch.from_numpy(nimgs).type(torch.cuda.FloatTensor)
            pred = model(inputs)
            test_pd = torch_util.threshold_tensor_batch(pred)

            all_gene.extend(genes)
            all_gt.extend(labels.astype(np.int))
            all_pd.extend(np.array(test_pd))

    df = pd.DataFrame({'gene': all_gene, 'gt': all_gt, 'pd': all_pd})
    df.to_csv("cancer.txt", header=True, index=False, sep=',')


def onehot(l):
    label = np.zeros(NUM_CLASSES, dtype=int)
    for i in l:
        label[i] = 1
    return label


def load_si_data(mode='cancer'):
    '''load single instance data'''
    if mode == 'cancer':
        tlist = tissue_list
        meta = META_DIR
    else:
        tlist = normal_tissue_list
        meta = NORMAL_META_DIR

    label_d = datautil.load_gene_label(size=1)
    d = {}
    for t in tlist:
        tissue_dir = os.path.join(meta, t)
        for gf in os.listdir(tissue_dir):
            gene = gf.split(".")[0]
            if gene not in label_d:
                continue
            gfp = os.path.join(tissue_dir, gf)
            with open(gfp, 'r') as f:
                for line in f.readlines():
                    pic = line.strip("\n")
                    d[pic] = {}
                    d[pic]['gene'] = gene
                    d[pic]['tissue'] = t
                    d[pic]['gt'] = onehot(label_d[gene])
    return d


def predict_si(mode='cancer'):
    '''per image predict'''
    if mode == 'cancer':
        train_data = load_si_data('cancer')
        result_pth = 'cancer_si.txt'
        fv_dir = CANCER_FV_DIR
    else:
        train_data = load_si_data('normal')
        result_pth = 'normal_si.txt'
        fv_dir = NORMAL_FV_DIR

    model_name = "ubploss/transformer_res18-128_size0_autoloss_alpha16"
    model_dir = os.path.join("./modeldir/%s" % model_name)
    model_pth = os.path.join(model_dir, "model.pth")

    model = torch.load(model_pth).cuda()
    model.eval()

    items = []
    for i, img in enumerate(train_data):
        pic = img
        gene = train_data[img]['gene']
        tissue = train_data[img]['tissue']
        gt = train_data[img]['gt']
        npy = os.path.join(fv_dir, gene, img.replace('jpg', 'npy'))
        if not os.path.exists(npy):
            continue
        nimgs = np.load(npy)
        inputs = torch.from_numpy(nimgs).type(torch.cuda.FloatTensor)
        out = model(inputs)
        prob = out.squeeze().data.cpu().numpy()
        pred = np.greater_equal(prob, [0.5] * 6).astype(int)
        # print("gt", gt)
        # print("pred", pred)
        items.append((pic, gene, tissue, prob, pred, gt))

    # print("items", items)
    pics, genes, ts, ps, preds, gts = zip(*items)
    df = pd.DataFrame({
        'img': pics, 'gene': genes, 'tissue': ts,
        'prob': ps, 'pred': preds, 'gt': gts})
    df = df.set_index('img')
    df.to_csv(result_pth, header=True, index=True, sep=',',
              columns=['gt', 'pred', 'prob', 'gene', 'tissue'])


if __name__ == "__main__":
    # predict()
    predict_si(mode='normal')
