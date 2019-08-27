#!/usr/bin/env python
# -*- coding: utf-8

import os
import random
import numpy as np
from util import datautil
from util import constant as c

DATA_DIR = c.QDATA_DIR
SUPP_DATA_DIR = c.SUPP_DATA_DIR
APPROVE_DATA_DIR = c.APPROVE_DATA_DIR

# tissue dir store img name for the 4 tissues
TISSUE_DIR = c.TISSUE_DIR
FV_DIR = c.FV_DIR

NUM_CLASSES = 6


def get_gene_pics(gene):
    pics = []
    for t in ['liver', 'breast', 'prostate', 'bladder']:
        tp = os.path.join(TISSUE_DIR, t, "%s.txt" % gene)
        if os.path.exists(tp):
            with open(tp, 'r') as f:
                pics.extend([l.strip("\n") for l in f.readlines()])
    return pics


def load_train_data(size=1, balance=False, fv='res18-128'):
    gene_list = datautil.get_train_gene_list(size)
    if balance:
        gene_list = datautil.get_balanced_gene_list(gene_list, size)
    return _load_data(gene_list, size=size, fv=fv)


def load_val_data(size=1, fv='res18-128'):
    gene_list = datautil.get_val_gene_list(size)
    return _load_data(gene_list, size=size, fv=fv)


def load_test_data(size=1, fv='res18-128'):
    gene_list = datautil.get_test_gene_list(size)
    return _load_data(gene_list, size=size, fv=fv)


def _handle_load(gene, d, fv='res18-128'):
    FV_DIR = os.path.join(c.ROOT, "enhanced_4tissue_fv", fv.replace('-', '_'))
    genef = os.path.join(FV_DIR, "%s.npy" % gene)
    nimg = np.load(genef)
    gene_label = np.zeros(NUM_CLASSES)
    for l in d[gene]:
        gene_label[l] = 1
    timestep = nimg.shape[0]
    return (gene, nimg, gene_label, timestep)


def _load_data(gene_list, size=1, fv='res18-128'):
    if size == 0:
        d = datautil.load_enhanced_label()
    elif size == 1:
        d = datautil.load_supported_label()
    else:
        d = datautil.load_approved_label()

    q = [x for x in gene_list if x in d and len(get_gene_pics(x))]

    return [_handle_load(x, d, fv) for x in q]


def shuffle(items):
    index = list(range(len(items)))
    random.shuffle(index)
    return [items[i] for i in index]


def shuffle_with_idx(items):
    idx = np.random.permutation(range(len(items)))
    sitems = [items[x] for x in idx]
    return sitems, idx


def batch_fv(items, batch=128):
    length = len(items)
    batched = [items[i:i+batch] for i in range(0, length, batch)]
    for batch in batched:
        (genes, nimgs, labels, timesteps) = zip(*batch)

        maxt = np.max(timesteps)
        pad_imgs = []
        for img in nimgs:
            pad = np.pad(img, [(0, maxt-img.shape[0]), (0, 0)],
                         mode='constant', constant_values=0)
            pad_imgs.append(pad)
        yield (np.array(genes), np.array(pad_imgs),
               np.array(labels), np.array(timesteps))


def test():
    train_data = load_train_data()

    # for item in train_data:
    #     print(item[0])
    #     print(item[1].shape)
    #     print(item[2])
    #     print(item[3])

    for batch_data in batch_fv(shuffle(train_data), 4):
        print(batch_data)


if __name__ == "__main__":
    test()
