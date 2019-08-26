#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import math
from util import constant as c

NUM_CLASSES = 6

label_map = {
    "Nuclear membrane": 0,
    "Cytoplasm": 1,
    "Vesicles": 2,
    "Mitochondria": 3,
    "Golgi Apparatus": 4,
    "Nucleoli": 0,
    "Plasma Membrane": 1,
    "Nucleoplasm": 0,
    "Endoplasmic Reticulum": 5
}

four_tissue_list = ['liver', 'breast', 'prostate', 'bladder']
all_tissue_list = os.listdir(c.ALL_TISSUE_DIR)


def get_gene_pics(gene, tissue_list=four_tissue_list):
    pics = []
    for t in tissue_list:
        tp = os.path.join(c.TISSUE_DIR, t, "%s.txt" % gene)
        if os.path.exists(tp):
            with open(tp, 'r') as f:
                pics.extend([l.strip("\n") for l in f.readlines()])
    return pics


def get_gene_list(size=1):
    '''not consider train/val/test'''
    gene_list = []
    if size >= 0:
        gene_list += get_enhanced_gene_list()
    if size >= 1:
        gene_list += get_supported_gene_list()
    if size >= 2:
        gene_list += get_approved_gene_list()

    return gene_list


def get_balanced_gene_list(gene_list, size=0):
    return get_global_balanced_gene_list(gene_list, size)


def _label_ratio(gene_list, gene_label):
    '''return label_ratio & label_gene_list for current gene_list'''
    total = len(gene_list)
    label_gene_list = [[] for _ in range(NUM_CLASSES)]
    for gene in gene_list:
        for l in gene_label[gene]:
            label_gene_list[l].append(gene)
    label_ratio = [len(x) / (total - len(x)) for x in label_gene_list]
    return label_ratio, label_gene_list


def get_global_balanced_gene_list(gene_list, size=0):
    '''consider label correlation for label balance'''
    gene_label = load_gene_label(size)
    gene_list = [gene for gene in gene_list if gene in gene_label and
                 len(get_gene_pics(gene)) > 0]

    ret_list = gene_list
    for i in range(100):
        ratio, label_gene_list = _label_ratio(ret_list, gene_label)
        # print("\n---------%d iteration----------" % i)
        # print("ratio", ratio)
        # print("count", [len(x) for x in label_gene_list])

        minratio = min(ratio)
        if minratio > 0.3:
            break
        minlabel = ratio.index(minratio)
        repeat_count = int(math.sqrt(1.0/minratio))
        repeat_gene = label_gene_list[minlabel] * repeat_count
        ret_list = ret_list + repeat_gene

    idx = np.random.permutation(np.array(range(len(ret_list))))
    ret_list = [ret_list[i] for i in idx]

    return ret_list


def get_local_balanced_gene_list(gene_list, size=0):
    '''consider label balance, only for train gene list'''
    gene_label = load_gene_label(size)
    gene_list = [gene for gene in gene_list if gene in gene_label and
                 len(get_gene_pics(gene)) > 0]
    # for each label, get its gene list
    label_gene_list = [[] for _ in range(NUM_CLASSES)]
    for gene in gene_list:
        for l in gene_label[gene]:
            label_gene_list[l].append(gene)
    label_freq = [len(x) for x in label_gene_list]
    maxfreq = max(label_freq)

    # repeat_count = [maxfreq // x for x in label_freq]
    repeat_count = [int(math.sqrt(maxfreq / x)) for x in label_freq]
    repeat_list = [l * r for l in label_gene_list for r in repeat_count]

    # extra_count = [maxfreq % x for x in label_freq]
    # extra_list = [l[:e] for l in label_gene_list for e in extra_count]
    # label_balance_list = [r + e for r in repeat_list for e in extra_list]

    label_balance_list = repeat_list
    ret_list = [l for label_list in label_balance_list for l in label_list]

    idx = np.random.permutation(np.array(range(len(ret_list))))
    ret_list = [ret_list[i] for i in idx]

    return ret_list


def get_enhanced_gene_list():
    '''some gene marked as enhanced but do not have enhanced label'''
    return [x for x in os.listdir(c.DATA_DIR)
            if len(os.listdir(os.path.join(c.DATA_DIR, x)))]


def get_supported_gene_list():
    return [x for x in os.listdir(c.SUPP_DATA_DIR)
            if len(os.listdir(os.path.join(c.SUPP_DATA_DIR, x)))]


def get_approved_gene_list():
    return [x for x in os.listdir(c.APPROVE_DATA_DIR)
            if len(os.listdir(os.path.join(c.APPROVE_DATA_DIR, x)))]


def load_gene_label(size=1):
    if size == 0:
        return load_enhanced_label()
    elif size == 1:
        return load_supported_label()
    else:
        return load_approved_label()


def load_enhanced_label():
    return _load_label_from_file("enhanced_label.txt")


def load_supported_label():
    return _load_label_from_file("supported_label.txt")


def load_approved_label():
    return _load_label_from_file("approved_label.txt")


def _load_label_from_file(fname):
    d = {}
    pardir = os.path.join(os.path.dirname(__file__), os.pardir)
    label_file = os.path.join(pardir, "label", fname)
    with open(label_file, 'r') as f:
        for line in f.readlines():
            gene, label = line.strip("\n").split("\t")
            labels = [label_map[x] for x in label.split(",") if x]
            if labels:
                d[gene] = labels
    return d


def get_train_gene_list(size=1, ratio=0.7):
    gene_list = []

    if size >= 2:
        approved = get_supported_gene_list()
        pivot = int(len(approved) * ratio)
        gene_list.extend(approved[:pivot])

    if size >= 1:
        supported = get_supported_gene_list()
        pivot = int(len(supported) * ratio)
        gene_list.extend(supported[:pivot])

    if size >= 0:
        enhanced = get_enhanced_gene_list()
        pivot = int(len(enhanced) * ratio)
        gene_list.extend(enhanced[:pivot])

    return gene_list


def get_val_gene_list(size=1, sratio=0.7, eratio=0.9):
    gene_list = []

    if size >= 2:
        approved = get_supported_gene_list()
        spivot = int(len(approved) * sratio)
        epivot = int(len(approved) * eratio)
        gene_list.extend(approved[spivot:epivot])

    if size >= 1:
        supported = get_supported_gene_list()
        spivot = int(len(supported) * sratio)
        epivot = int(len(supported) * eratio)
        gene_list.extend(supported[spivot:epivot])

    if size >= 0:
        enhanced = get_enhanced_gene_list()
        spivot = int(len(enhanced) * sratio)
        epivot = int(len(enhanced) * eratio)
        gene_list.extend(enhanced[spivot:epivot])

    return gene_list


def get_test_gene_list(size=1, sratio=0.9):
    gene_list = []

    if size >= 2:
        approved = get_supported_gene_list()
        spivot = int(len(approved) * sratio)
        gene_list.extend(approved[spivot:])

    if size >= 1:
        supported = get_supported_gene_list()
        spivot = int(len(supported) * sratio)
        gene_list.extend(supported[spivot:])

    if size >= 0:
        enhanced = get_enhanced_gene_list()
        spivot = int(len(enhanced) * sratio)
        gene_list.extend(enhanced[spivot:])

    return gene_list


def shuffle(items, batch=128):
    '''shuffle & split into batch'''
    length = len(items)
    index = list(range(len(items)))
    random.shuffle(index)
    a = [items[i] for i in index]
    return [a[i:i+batch] for i in range(0, length, batch)]


def get_label_freq(size=1):
    d = load_gene_label(size)
    train_list = [x for x in get_train_gene_list(size)
                  if x in d and len(get_gene_pics(x)) > 0]
    ntrain = len(train_list)

    def onehot(l):
        label = np.zeros(NUM_CLASSES)
        for i in l:
            label[i] = 1
        return label

    train_label = np.array([onehot(d[x]) for x in train_list])
    nlabel = np.sum(train_label, axis=0)

    return np.array(nlabel / ntrain)


def label_stat(size=1):
    print("------label stat for size=%d dataset-----\n" % size)
    d = load_gene_label(size)

    train_list = [x for x in get_train_gene_list(size)
                  if x in d and len(get_gene_pics(x)) > 0]
    val_list = [x for x in get_val_gene_list(size)
                if x in d and len(get_gene_pics(x)) > 0]
    test_list = [x for x in get_test_gene_list(size)
                 if x in d and len(get_gene_pics(x)) > 0]

    print("train num gene:", len(train_list))
    print("val num gene:", len(val_list))
    print("test num gene:", len(test_list))

    def onehot(l):
        label = np.zeros(NUM_CLASSES)
        for i in l:
            label[i] = 1
        return label

    train_label = np.array([onehot(d[x]) for x in train_list])
    val_label = np.array([onehot(d[x]) for x in val_list])
    test_label = np.array([onehot(d[x]) for x in test_list])

    uniq, count = np.unique(np.sum(train_label, axis=1), return_counts=True)
    label_count = dict(zip([int(x) for x in uniq], count))
    print("label count", label_count)

    print("train count", np.sum(train_label, axis=0))
    print("val count", np.sum(val_label, axis=0))
    print("test count", np.sum(test_label, axis=0))

    print("train cardinality", np.mean(np.sum(train_label, axis=1)))
    print("val cardinality", np.mean(np.sum(val_label, axis=1)))
    print("test cardinality", np.mean(np.sum(test_label, axis=1)))
    print("\n")


def test_global_balance(size=0):
    d = load_gene_label(size)

    train_list = [x for x in get_train_gene_list(size)
                  if x in d and len(get_gene_pics(x)) > 0]

    local_list = get_local_balanced_gene_list(train_list)
    global_list = get_global_balanced_gene_list(train_list)

    origin_ratio, _ = _label_ratio(train_list, d)
    local_ratio, _ = _label_ratio(local_list, d)
    global_ratio, _ = _label_ratio(global_list, d)
    print("----test global balance----")
    print("origin", origin_ratio)
    print("local", local_ratio)
    print("global", global_ratio)


def count_img():
    train_list = get_train_gene_list(size=0)
    val_list = get_val_gene_list(size=0)
    test_list = get_test_gene_list(size=0)

    tissue_datadir = '/ndata/longwei/hpa/tissuedata'
    tissue_list = ['bladder', 'breast', 'liver', 'prostate']

    train_count = 0
    val_count = 0
    test_count = 0

    for t in tissue_list:
        tdir = os.path.join(tissue_datadir, t)

        for g in train_list:
            gp = os.path.join(tdir, "%s.txt" % g)
            if not os.path.exists(gp):
                continue
            with open(gp, 'r') as f:
                c = len(f.readlines())
                train_count += c

        for vg in val_list:
            gp = os.path.join(tdir, "%s.txt" % g)
            if not os.path.exists(gp):
                continue
            with open(gp, 'r') as f:
                c = len(f.readlines())
                val_count += c

        for tg in test_list:
            gp = os.path.join(tdir, "%s.txt" % g)
            if not os.path.exists(gp):
                continue
            with open(gp, 'r') as f:
                c = len(f.readlines())
                test_count += c

    print('train gene count', len(train_list))
    print("train img count", train_count)

    print('val gene count', len(val_list))
    print("val img count", val_count)

    print('test gene count', len(test_list))
    print("test img count", test_count)


if __name__ == "__main__":
    # label_stat(0)
    # label_stat(1)
    # label_stat(2)
    # print(get_label_freq(2))
    # test_global_balance()
    count_img()
