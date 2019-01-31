#!/usr/bin/env python
# -*- coding: utf-8 -*-

# load liver data

import os
import random
import numpy as np


label_map = {
    "Cytoplasm": 0,
    "Mitochondria": 1,
    "Focal Adhesions": 2,
    "Centrosome": 2,
    "Microtubule organizing center": 2,
    "Cytoskeleton (Microtubules)": 2,
    "Cytoskeleton (Intermediate filaments)": 2,
    "Cytoskeleton (Actin filaments)": 2,
    "Nucleoli": 2,
    "Nucleus": 2,
    "Nuclear membrane": 2,
    "Nucleus but not nucleoli": 2,
    "Cell Junctions": 2,
    "Plasma membrane": 2,
    "Vesicles": 3,
    "Endoplasmic reticulum": 4,
    "Golgi apparatus": 5,
}

NUM_CLASSES = 6

LABEL_PATH = '/ndata/longwei/hpa/liver_label.txt'
LIVER_DIR = '/ndata/longwei/hpa/liver/'


def load_gene_label():
    d = {}
    with open(LABEL_PATH, 'r') as f:
        for line in f.readlines():
            gene, label = line.strip().split(":")
            if not os.path.exists(os.path.join(LIVER_DIR, gene)):
                continue
            label = eval(label).split(",")
            d[gene] = [0] * NUM_CLASSES
            for l in label:
                d[gene][label_map[l]] = 1
    return d


def split_train_val_test(li):
    num = len(li)
    ntrain = int(num * 0.6)
    nval = int(num * 0.8)
    return li[:ntrain], li[ntrain:nval], li[nval:]


def load_liver_fv_train(batch=32):
    datadir = "/ndata/longwei/hpa/liverfv/res18"
    gene_label_map = load_gene_label()
    genefs = [g for g in os.listdir(datadir)
              if os.path.splitext(g)[0] in gene_label_map.keys()]

    train, val, test = split_train_val_test([os.path.join(datadir, x)
                                            for x in genefs])
    random.shuffle(train)
    return _load_img_fv(train, batch)


def load_liver_fv_val():
    datadir = "/ndata/longwei/hpa/liverfv/res18"
    gene_label_map = load_gene_label()
    genefs = [g for g in os.listdir(datadir)
              if os.path.splitext(g)[0] in gene_label_map.keys()]

    train, val, test = split_train_val_test([os.path.join(datadir, x)
                                            for x in genefs])
    batch = len(val)
    return _load_img_fv(val, batch)


def load_liver_fv_test():
    datadir = "/ndata/longwei/hpa/liverfv/res18"
    gene_label_map = load_gene_label()
    genefs = [g for g in os.listdir(datadir)
              if os.path.splitext(g)[0] in gene_label_map.keys()]

    train, val, test = split_train_val_test([os.path.join(datadir, x)
                                            for x in genefs])
    batch = len(test)
    return _load_img_fv(test, batch)


def _load_img_fv(genefs, batch):
    ret = []
    gene_label_map = load_gene_label()
    chunks = [genefs[i:i+batch] for i in range(0, len(genefs), batch)]
    for chunk in chunks:
        batch_size = len(chunk)
        chunk_imgs = []
        chunk_labels = []
        timesteps = []
        max_timestep = 0
        for genef in chunk:
            gene = os.path.basename(os.path.splitext(genef)[0])

            # for liver, already one hot
            chunk_labels.append(gene_label_map[gene])

            img = np.load(genef)
            nimg = np.expand_dims(img, axis=1)
            # nimg = nimg[:10, :, :]
            timesteps.append(nimg.shape[0])
            chunk_imgs.append(nimg)

        max_timestep = np.max(timesteps)
        imgs = []
        for ci in chunk_imgs:
            pad = np.pad(ci, [(0, max_timestep-ci.shape[0]), (0, 0), (0, 0)],
                         mode='constant', constant_values=0)
            imgs.append(pad)

        imgs = np.concatenate(imgs, axis=1)
        # labels = np.concatenate(chunk_labels, axis=0)
        labels = np.array(chunk_labels)

        # timesteps = [i if i < 30 else 30 for i in timesteps]
        # yield imgs, labels, batch_size, timesteps

        ret.append((imgs, labels, batch_size, timesteps))
    return ret


if __name__ == "__main__":
    d = load_gene_label()
    # print(len(d))
    print("train")
    for item in load_liver_fv_train():
        print(item[1], item[3])
    print("test")
    # for item in load_liver_fv_test():
    #     print(item[1])
