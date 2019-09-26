#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
import threading
import json
import queue
import psutil
import time
from util import datautil
from util import constant as c

roiSize = 224
# roiSize = 60

DATA_DIR = c.DATA_DIR
PATCH_DIR = c.PATCH_DIR


def _handle_extract_pt(q, roi):
    while True:
        item = q.get()
        if not item:
            break
        gene, img = item
        imgpath = os.path.join(DATA_DIR, gene, img)
        nimg = cv2.imread(imgpath)
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        nimg = np.transpose(nimg, (2, 0, 1))
        nimg = nimg.astype('float')

        gene_dir = os.path.join(PATCH_DIR, gene)
        try:
            if not os.path.exists(gene_dir):
                os.mkdir(gene_dir)
        except Exception:
            pass
        imgpt = os.path.join(gene_dir, "%s.pt.npy" % img)
        if os.path.exists(imgpt):
            print("already extract for %s" % imgpt)
            continue

        points = list(roi[gene][img])
        for i, point in enumerate(points):
            xc, yc = [int(x) for x in point]
            xs = xc - roiSize // 2
            xe = xc + roiSize // 2
            ys = yc - roiSize // 2
            ye = yc + roiSize // 2

            patch = nimg[:, xs:xe, ys:ye]
            imgpt = os.path.join(gene_dir, "%s.%d.npy" % (img, i))
            np.save(imgpt, patch)

        time.sleep(0.1)


def _handle_extract(q, roi):
    while True:
        item = q.get()
        if not item:
            break
        gene, img = item
        imgpath = os.path.join(DATA_DIR, gene, img)
        nimg = cv2.imread(imgpath)

        gene_dir = os.path.join(PATCH_DIR, gene)
        try:
            if not os.path.exists(gene_dir):
                os.mkdir(gene_dir)
        except Exception:
            pass

        points = list(roi[gene][img])
        for i, point in enumerate(points):
            xc, yc = [int(x) for x in point]
            xs = xc - roiSize // 2
            xe = xc + roiSize // 2
            ys = yc - roiSize // 2
            ye = yc + roiSize // 2

            patch = nimg[xs:xe, ys:ye, :]
            imgpt = os.path.join(gene_dir, "%s.%d.jpg" % (img, i))
            cv2.imwrite(imgpt, patch)

        time.sleep(0.1)


def extract(roiPath="roi/roi%s.json" % roiSize):
    gene_label = datautil.load_gene_label(0)
    all_genes = datautil.get_gene_list(0)
    all_genes = [gene for gene in all_genes if gene in gene_label]

    with open(roiPath, 'r') as f:
        roi = json.load(f)

    gene_imgs = [(gene, img) for gene in all_genes
                 for img in roi[gene].keys()]

    q = queue.Queue()

    for item in gene_imgs:
        q.put(item)

    if not os.path.exists(PATCH_DIR):
        os.mkdir(PATCH_DIR)

    nworker = psutil.cpu_count()
    jobs = []
    for i in range(nworker):
        p = threading.Thread(target=_handle_extract, args=(q, roi))
        jobs.append(p)
        p.daemon = True
        p.start()
        q.put(None)

    for j in jobs:
        j.join()


if __name__ == "__main__":
    extract()
