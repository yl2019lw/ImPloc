#!/usr/bin/env python
# -*- coding: utf-8 -*-

# convert label, use the result of parseLabel.py
# eg. generate enhanced_label.txt from enhanced.txt

import os
from cellDict import cellDict

# files = ['enhanced.txt', 'supported.txt']
files = ['approved.txt']


def get_gene_label(fpath='supported_2000.txt'):
    gene_label = {}
    with open(os.path.join(os.path.curdir, fpath), "r") as f:
        f.readline()
        for line in f.readlines():
            gene, labels = line.strip("\n").split(",")
            labels = [cellDict[x.lower()] for x in labels.split(";") if x]
            gene_label[gene] = list(set(labels))

    base, ext = os.path.splitext(fpath)
    opath = "%s_label%s" % (base, ext)
    with open(opath, 'w') as f:
        for gene, labels in gene_label.items():
            line = gene + "\t" + ",".join(labels) + "\n"
            f.write(line)
    return gene_label


def all_occurred_labels(fpath='uncertain_2000.txt'):
    all_labels = []
    with open(os.path.join(os.path.curdir, fpath), "r") as f:
        f.readline()
        for line in f.readlines():
            _, labels = line.strip("\n").split(",")
            labels = [cellDict[x.lower()] for x in labels.split(";") if x]
            all_labels.extend(set(labels))
    all_labels = set(all_labels)
    with open(os.path.join(os.path.curdir, "all_labels.txt"), "w") as f:
        for l in all_labels:
            f.write(l)
            f.write("\n")
    return all_labels


if __name__ == "__main__":
    for fp in files:
        get_gene_label(fp)
    # gl = get_gene_label("supported_2000.txt")
    # print(gl)
    # al = all_occurred_labels("supported_2000.txt")
    # print(al)
