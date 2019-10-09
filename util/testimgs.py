#!/usr/bin/env python
# -*- coding: utf-8 -*-

# independent test from iLocator

import os
import numpy as np
import math
import shutil
from util import datautil
from util import constant as c
from util import npmetrics


def copy_testimgs():
    DST = "/tmp/testimgs"

    test_genes = datautil.get_test_gene_list(size=0)
    for g in test_genes:
        pics = datautil.get_gene_pics(g)
        dst_dir = os.path.join(DST, g)
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)
        for pic in pics:
            src = os.path.join(c.QDATA_DIR, g, pic)
            dst = os.path.join(dst_dir, pic)
            shutil.copy(src, dst)


def evaluate_ilocator():
    test_genes = datautil.get_test_gene_list(size=0)
    pd = []
    gt = []
    d = datautil.load_enhanced_label()

    for g in test_genes:
        pics = datautil.get_gene_pics(g)
        g_scores = []
        for pic in [x.replace(".", "_") for x in pics]:
            spath = os.path.abspath(
                os.path.join("util/testimgs_ilocator_result/%s.txt" % pic))
            with open(spath, 'r') as f:
                score = [float(x) for x in f.readline().strip().split()]
                if any([math.isnan(x) for x in score]):
                    pass
                else:
                    g_scores.append(score)
        g_scores = np.stack(g_scores)
        g_scores = np.mean(g_scores, axis=0)
        pd.append(g_scores)

        gene_label = np.zeros(6)
        for l in d[g]:
            gene_label[l] = 1
        gt.append(gene_label)

    gt = np.stack(gt)

    pd = np.stack(pd)
    # rearrange label order
    idx = np.array([5, 0, 6, 4, 2, 1])
    pd = pd[:, idx]

    thr = pd.max(axis=1)
    zeros = np.zeros(thr.shape[0])
    thr = np.min(np.stack([zeros, thr], axis=1), axis=1)

    pd = np.greater_equal(pd, thr[:, np.newaxis]).astype(int)
    npmetrics.write_metrics(gt, pd, "util/ilocator.txt")


if __name__ == "__main__":
    evaluate_ilocator()
