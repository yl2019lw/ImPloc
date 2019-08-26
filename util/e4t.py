#!/usr/bin/env python
# -*- coding: utf-8 -*-

# select img: enhanced, 4 tissue, high quality from qdata

import os
import shutil
import threading
import queue
from util import datautil
from util import constant as c

DATA_DIR = c.QDATA_DIR
TARGET = c.DATA_DIR


def copy(q):
    while True:
        item = q.get()
        if item is None:
            break
        src_p, tgt_p = item
        shutil.copy(src_p, tgt_p)
        q.task_done()


def run():
    q = queue.Queue()
    all_genes = datautil.get_gene_list(0)
    gene_label = datautil.load_gene_label(0)
    all_genes = [gene for gene in all_genes if gene in gene_label]

    for gene in all_genes:
        src_dir = os.path.join(DATA_DIR, gene)
        tgt_dir = os.path.join(TARGET, gene)
        if not os.path.exists(tgt_dir) and os.listdir(src_dir):
            os.mkdir(tgt_dir)
        for img in datautil.get_gene_pics(gene):
            src_p = os.path.join(src_dir, img)
            tgt_p = os.path.join(tgt_dir, img)
            if not os.path.exists(tgt_p):
                q.put((src_p, tgt_p))

    jobs = []
    NUM_THREADS = 20
    for i in range(NUM_THREADS):
        p = threading.Thread(target=copy, args=(q,))
        jobs.append(p)
        p.start()

    q.join()

    for i in range(NUM_THREADS):
        q.put(None)

    for j in jobs:
        j.join()


def clear():
    all_genes = datautil.get_gene_list(0)
    gene_label = datautil.load_gene_label(0)
    all_genes = [gene for gene in all_genes if gene in gene_label]

    for gene in all_genes:
        tgt_dir = os.path.join(TARGET, gene)
        if os.path.exists(tgt_dir) and not os.listdir(tgt_dir):
            os.rmdir(tgt_dir)


if __name__ == "__main__":
    # run()
    clear()
