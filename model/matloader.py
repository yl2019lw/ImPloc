#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from scipy import io
from util import datautil
from util import constant as c

MATLAB_FV_DIR = c.MATLAB_FV_DIR

NUM_CLASSES = 6


def load_train_data(size=0, balance=False, fv='matlab'):
    gene_list = datautil.get_train_gene_list(size)
    if balance:
        gene_list = datautil.get_balanced_gene_list(gene_list, size)
    return _load_data(gene_list, size=size)


def load_val_data(size=0, fv='matlab'):
    gene_list = datautil.get_val_gene_list(size)
    return _load_data(gene_list, size=size)


def load_test_data(size=0, fv='matlab'):
    gene_list = datautil.get_test_gene_list(size)
    return _load_data(gene_list, size=size)


def kfold_split(fold=1):
    all_genes = datautil.get_enhanced_gene_list()
    candidate_genes = all_genes[:int(len(all_genes) * 0.9)]
    cl = len(candidate_genes)
    vl = cl // 10
    val_start = vl * (fold - 1)
    val_end = vl * fold
    val_genes = set(candidate_genes[val_start:val_end])
    train_genes = set(candidate_genes) - set(val_genes)
    val_genes = list(val_genes)
    train_genes = list(train_genes)

    return train_genes, val_genes


def load_kfold_train_data(fold=1, fv='matlab'):
    train_genes, val_genes = kfold_split(fold)
    return _load_data(train_genes, size=0)


def load_kfold_val_data(fold=1, fv='matlab'):
    train_genes, val_genes = kfold_split(fold)
    return _load_data(val_genes, size=0)


def load_kfold_test_data(fold=1, fv='matlab'):
    return load_test_data(size=0)


def _handle_load(gene, d):
    gene_dir = os.path.join(MATLAB_FV_DIR, gene)
    gene_fv = []
    for matf in os.listdir(gene_dir):
        fv = io.loadmat(os.path.join(gene_dir, matf))['features']
        if np.isnan(fv).any():
            # print("find nan in ", gene, matf)
            continue
        gene_fv.append(fv)
    gene_fv = np.concatenate(gene_fv)
    gene_label = np.zeros(NUM_CLASSES)
    for l in d[gene]:
        gene_label[l] = 1
    timestep = gene_fv.shape[0]
    return (gene, gene_fv, gene_label, timestep)


def _load_data(gene_list, size=0):
    if size == 0:
        d = datautil.load_enhanced_label()
    elif size == 1:
        d = datautil.load_supported_label()
    else:
        d = datautil.load_approved_label()

    q = [x for x in gene_list if x in d and len(datautil.get_gene_pics(x))]
    q = [x for x in q if os.path.exists(os.path.join(MATLAB_FV_DIR, x))]

    return [_handle_load(x, d) for x in q]


def shuffle(items):
    index = list(range(len(items)))
    np.random.shuffle(index)
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

    for item in train_data:
        print(item[0])
        print(item[1].shape)
        print(item[2])
        print(item[3])

    # for batch_data in batch_fv(shuffle(train_data), 4):
    #     print(batch_data)


if __name__ == "__main__":
    test()
