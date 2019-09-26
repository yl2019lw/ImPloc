#!/usr/bin/env python
# -*- coding: utf-8 -*-

# extract ROI center coordinates from Imgs

import os
import threading
import queue
import json
import cv2
import numpy as np
from util import datautil
from util import constant as c

DATA_DIR = c.DATA_DIR

roiSize = 224
# roiSize = 60


def do_extract(imgpath):
    points = []
    window = roiSize
    stride = roiSize // 2
    # stride = roiSize
    # size = 3000

    count = 0
    img = cv2.imread(imgpath)
    h, w, c = img.shape
    nh = h - h % window
    nw = w - w % window
    sh = (h % window) // 2
    sw = (w % window) // 2
    img = img[sh:sh+nh, sw:sw+nw, :]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    def valid_hue(hue):
        # lhue = 5.0
        # hhue = 25.0
        # lp = np.percentile(hue, 25)
        # hp = np.percentile(hue, 75)
        lhue = 10.0
        hhue = 20.0
        lp = np.percentile(hue, 10)
        hp = np.percentile(hue, 90)
        if lp < lhue or lp > hhue:
            return False
        if hp < lhue or hp > hhue:
            return False

        return True

    for i in range(0, nh, stride):
        for j in range(0, nw, stride):
            spatch = s[i:i+window, j:j+window]
            hpatch = h[i:i+window, j:j+window]
            if not spatch.shape == (window, window):
                continue

            if not hpatch.shape == (window, window):
                continue

            if np.mean(spatch) < 30:
                continue

            if not valid_hue(hpatch):
                continue

            xc = i + int(window // 2)
            yc = j + int(window // 2)
            # print("append %s:(%d, %d)" % (imgpath, xc, yc))
            points.append((xc, yc))
            count = count + 1
    print("%s/%s:%d" % (imgpath.split("/")[-2], imgpath.split("/")[-1], count))
    return points


def extract_img(q, outq):
    while True:
        item = q.get()
        if item is None:
            break
        gene, img = item
        imgpath = os.path.join(DATA_DIR, gene, img)
        points = do_extract(imgpath)
        outq.put((gene, img, points))
        q.task_done()


def extract():
    gene_label = datautil.load_gene_label(0)
    all_genes = datautil.get_gene_list(0)
    all_genes = [gene for gene in all_genes if gene in gene_label]

    q = queue.Queue()
    outq = queue.Queue()
    d_points = {}

    for gene in all_genes:
        d_points[gene] = {}
        for img in datautil.get_gene_pics(gene):
            q.put((gene, img))

    NUM_THREADS = 20

    jobs = []
    for i in range(NUM_THREADS):
        p = threading.Thread(target=extract_img, args=(q, outq))
        jobs.append(p)
        p.start()

    q.join()

    for i in range(NUM_THREADS):
        q.put(None)

    for j in jobs:
        j.join()

    while not outq.empty():
        gene, img, points = outq.get()
        d_points[gene][img] = points

    with open("roi/roi%d.json" % roiSize, "w") as f:
        json.dump(d_points, f)


if __name__ == "__main__":
    extract()
