#!/usr/bin/env python
# -*- coding: utf-8

# just a copy of fvloader

import os
import random
import numpy as np
from util import datautil
from util import constant as c

FV_DIR = c.CANCER_FV_DIR
NUM_CLASSES = 6


def load_cancer_data():
    gene_list = [x.split(".")[0] for x in os.listdir(FV_DIR)]
    return _load_data(gene_list, size=0)


def _handle_load(gene, d):
    genef = os.path.join(FV_DIR, "%s.npy" % gene)
    nimg = np.load(genef)
    gene_label = np.zeros(NUM_CLASSES)
    for l in d[gene]:
        gene_label[l] = 1
    timestep = nimg.shape[0]
    return (gene, nimg, gene_label, timestep)


def _load_data(gene_list, size=1):
    if size == 0:
        d = datautil.load_enhanced_label()
    elif size == 1:
        d = datautil.load_supported_label()
    else:
        d = datautil.load_approved_label()

    q = [x for x in gene_list if x in d]

    return [_handle_load(x, d) for x in q]


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
    train_data = load_cancer_data()

    # for item in train_data:
    #     print(item[0])
    #     print(item[1].shape)
    #     print(item[2])
    #     print(item[3])

    for batch_data in batch_fv(shuffle(train_data), 4):
        print(batch_data)


if __name__ == "__main__":
    test()
