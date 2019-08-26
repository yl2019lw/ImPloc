#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import queue
import threading
import numpy as np
import random
import cv2
import time
from util import datautil
from util import constant as c

# as enhanced & supported data at different location
DATA_DIR = c.QDATA_DIR
SUPP_DATA_DIR = c.SUPP_DATA_DIR
APPROVE_DATA_DIR = c.APPROVE_DATA_DIR
# tissue dir store img name for the 4 tissues
TISSUE_DIR = c.TISSUE_DIR

NUM_CLASSES = 6


def get_gene_pics(gene):
    pics = []
    for t in ['liver', 'breast', 'prostate', 'bladder']:
        tp = os.path.join(TISSUE_DIR, t, "%s.txt" % gene)
        if os.path.exists(tp):
            with open(tp, 'r') as f:
                pics.extend([l.strip("\n") for l in f.readlines()])
    return pics


def load_train_data(batch=128, size=1):
    gene_list = datautil.get_train_gene_list(size)
    random.shuffle(gene_list)

    return _load_data(gene_list, batch=batch, size=size)


def load_val_data(size=1):
    gene_list = datautil.get_val_gene_list(size)

    return _load_data(gene_list, isTrain=False, size=size)


def load_test_data(size=1):
    gene_list = datautil.get_test_gene_list(size)

    return _load_data(gene_list, isTrain=False, size=size)


def _handle_load(q, outq):
    d = datautil.load_gene_label(size=2)
    while not q.empty():
        gene = q.get()
        gene_img = []
        gene_label = np.zeros(NUM_CLASSES)
        for l in d[gene]:
            gene_label[l] = 1

        # some gene marked as enhanced but no enhance level label
        gene_dir = os.path.join(DATA_DIR, gene)
        if not os.path.exists(gene_dir):
            gene_dir = os.path.join(SUPP_DATA_DIR, gene)

        if not os.path.exists(gene_dir):
            gene_dir = os.path.join(APPROVE_DATA_DIR, gene)

        for pic in get_gene_pics(gene):
            image = os.path.join(gene_dir, pic)
            try:
                img = cv2.imread(image)
                # img = cv2.resize(img, (3000, 3000),
                #                  interpolation=cv2.INTER_CUBIC)
                img = cv2.resize(img, (224, 224),
                                 interpolation=cv2.INTER_CUBIC)
            except Exception as e:
                print("exception for image", image)
                print(e)

            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0)
            gene_img.append(img)
        if gene_img:
            gene_img = np.concatenate(np.array(gene_img), axis=0)
            outq.put((gene, gene_img, gene_label))


def _load_data(gene_list, batch=128, isTrain=True, size=1):
    q = queue.Queue()
    outq = queue.Queue()
    ngene = 0
    for gene in gene_list:
        # some gene marked as enhanced but no enhance level label
        if size == 0:
            d = datautil.load_enhanced_label()
            if gene not in d:
                continue
        elif size == 1:
            d = datautil.load_supported_label()
            if gene not in d:
                continue

        if not len(get_gene_pics(gene)):
            continue

        q.put(gene)
        ngene += 1

    print("actual ngene:", ngene, "isTrain:", isTrain, "data size:", size)

    jobs = []
    for i in range(10):
        t = threading.Thread(target=_handle_load, args=(q, outq))
        t.daemon = True
        jobs.append(t)
        t.start()

    items = []
    nemit = 0
    remain = ngene
    if not isTrain:
        batch = ngene
    while remain > 0:
        # print("remain", remain, q.qsize(), outq.qsize())
        remain = ngene - nemit * batch
        if remain > batch:
            quota = batch
        else:
            quota = remain
        if outq.qsize() >= quota and remain > 0:
            for i in range(quota):
                item = outq.get()
                items.append(item)
            yield items
            nemit += 1
            items = []

        time.sleep(1)

    for j in jobs:
        j.join()


def test():
    for item in load_test_data(size=2):
        print("----------batch----------------", len(item))
        for gene, img, label in item:
            print(gene, label, img.shape)


if __name__ == "__main__":
    test()
