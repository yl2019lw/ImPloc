#!/usr/bin/env python
# -*- coding: utf-8 -*-

# separte fv for each img form bag.

import os
import queue
import threading
import numpy as np
import sys
sys.path.append("../")
from util import datautil


BAG_FV_DIR = '/ndata/longwei/hpa/tissuefv/res18_128'
INS_FV_DIR = '/ndata/longwei/hpa/normalfv_all'

TISSUE_DIR = "/ndata/longwei/hpa/tissuedata"


def get_gene_pics(gene):
    pics = []
    for t in ['liver', 'breast', 'prostate', 'bladder']:
        tp = os.path.join(TISSUE_DIR, t, "%s.txt" % gene)
        if os.path.exists(tp):
            with open(tp, 'r') as f:
                pics.extend([l.strip("\n") for l in f.readlines()])
    return pics


def separate_image_fv(q, i):

    while True:

        gene = q.get()
        if gene is None:
            break

        src_npy_pth = os.path.join(BAG_FV_DIR, "%s.npy" % gene)
        nimg = np.load(src_npy_pth)

        pics = get_gene_pics(gene)
        if nimg.shape[0] != len(pics):
            print("size not match for %s" % gene, nimg.shape, len(pics))

        gene_dir = os.path.join(INS_FV_DIR, gene)
        if not os.path.exists(gene_dir):
            os.mkdir(gene_dir)

        for i, pic in enumerate(pics):
            tgt_npy = os.path.join(gene_dir, pic.replace('jpg', 'npy'))
            np.save(tgt_npy, np.expand_dims(nimg[i], axis=0))

        q.task_done()


def do_separate(genes):
    q = queue.Queue()
    for gene in genes:
        q.put(gene)

    NUM_THREADS = 20

    jobs = []
    for i in range(NUM_THREADS):
        p = threading.Thread(target=separate_image_fv, args=(q, i))
        jobs.append(p)
        p.start()

    q.join()

    for i in range(NUM_THREADS):
        q.put(None)

    for j in jobs:
        j.join()


def separate():
    gene_list = datautil.get_gene_list(size=0)
    do_separate(gene_list)


if __name__ == "__main__":
    separate()
