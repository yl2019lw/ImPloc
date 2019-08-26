#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision
import os
import cv2
import numpy as np
import time
import gpustat
import threading
import queue
from util import datautil
from util import constant as c

# for tissue data
DATA_DIR = c.QDATA_DIR
SUPP_DATA_DIR = c.SUPP_DATA_DIR
APPROVE_DATA_DIR = c.APPROVE_DATA_DIR
TISSUE_DIR = c.TISSUE_DIR

ALL_TISSUE_DIR = c.ALL_TISSUE_DIR
FV_BASE_DIR = c.ALL_FV_DIR


def get_gpu_usage(device=1):
    gpu_stats = gpustat.new_query()
    item = gpu_stats.jsonify()["gpus"][device]
    return item['memory.used'] / item['memory.total']


def extract_image_fv(q, model, i):

    def _extract_image(image):

        img = cv2.imread(image)
        img = cv2.resize(img, (3000, 3000), interpolation=cv2.INTER_CUBIC)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        inputs = torch.from_numpy(img).type(torch.cuda.FloatTensor)
        pd = model(inputs)
        return pd

    while True:

        while get_gpu_usage() > 0.9:
            print("---gpu full---", get_gpu_usage())
            time.sleep(1)
            torch.cuda.empty_cache()

        gene = q.get()
        if gene is None:
            break
        print("---extract -----", gene, q.qsize(), i)
        gene_dir = os.path.join(DATA_DIR, gene)
        if not os.path.exists(gene_dir):
            gene_dir = os.path.join(SUPP_DATA_DIR, gene)

        if not os.path.exists(gene_dir):
            gene_dir = os.path.join(APPROVE_DATA_DIR, gene)

        outpath = os.path.join(model.fvdir, "%s.npy" % gene)
        if os.path.exists(outpath):
            print("------already extracted---------", gene)
            q.task_done()
            continue

        # pds = [_extract_image(os.path.join(gene_dir, p))
        #        for p in datautil.get_gene_pics(gene)
        #        if os.path.splitext(p)[-1] == ".jpg"]

        pds = [_extract_image(os.path.join(gene_dir, p))
               for p in datautil.get_gene_pics(gene, datautil.all_tissue_list)
               if os.path.splitext(p)[-1] == ".jpg"]
        if pds:
            value = np.concatenate(pds, axis=0)
            print("----save-----", outpath)
            np.save(outpath, value)

        q.task_done()


def do_extract(model, genes):
    q = queue.Queue()
    for gene in genes:
        q.put(gene)

    for param in model.parameters():
        param.requires_grad = False

    NUM_THREADS = 8

    jobs = []
    for i in range(NUM_THREADS):
        p = threading.Thread(target=extract_image_fv, args=(q, model, i))
        jobs.append(p)
        p.start()

    q.join()

    for i in range(NUM_THREADS):
        q.put(None)

    for j in jobs:
        j.join()


class Extractor(nn.Module):

    def __init__(self, base="res18", dim=512):
        super(Extractor, self).__init__()
        self.base = base
        self.dim = dim
        if base == "res18":
            origin = torchvision.models.resnet18(pretrained=True)
            if dim == 512:
                self.fvdir = os.path.join(FV_BASE_DIR, "res18_512")
                self.end_layer = -2

            elif dim == 256:
                self.fvdir = os.path.join(FV_BASE_DIR, "res18_256")
                self.end_layer = -3

            elif dim == 128:
                self.fvdir = os.path.join(FV_BASE_DIR, "res18_128")
                self.end_layer = -4

            elif dim == 64:
                self.fvdir = os.path.join(FV_BASE_DIR, "res18_64")
                self.end_layer = -5

            else:
                raise Exception("invalid dim %d for extractor", dim)
            self.features = torch.nn.Sequential(
                *list(origin.children())[:self.end_layer])

        elif base == "vgg11":
            origin = torchvision.models.vgg11(pretrained=True)
            if dim == 512:
                self.fvdir = os.path.join(FV_BASE_DIR, "vgg11_512")
                self.end_layer = 15

            elif dim == 256:
                self.fvdir = os.path.join(FV_BASE_DIR, "vgg11_256")
                self.end_layer = 10

            elif dim == 128:
                self.fvdir = os.path.join(FV_BASE_DIR, "vgg11_128")
                self.end_layer = 5

            elif dim == 64:
                self.fvdir = os.path.join(FV_BASE_DIR, "vgg11_64")
                self.end_layer = 2

            else:
                raise Exception("invalid dim %d for extractor", dim)

            self.features = origin.features[:self.end_layer]
        else:
            raise Exception("invalid base %s for extractor", base)

        if not os.path.exists(self.fvdir):
            os.mkdir(self.fvdir)

        for p in self.features.parameters():
            p.require_grad = False

    def forward(self, x):
        f = self.features(x)
        f = torch.nn.AdaptiveAvgPool2d(1)(f)
        return f.view(f.size(0), -1)


def extract(base="res18", dim=128, size=1):
    model = Extractor(base, dim)
    model.share_memory()
    model.cuda()

    gene_list = datautil.get_gene_list(size)
    do_extract(model, gene_list)


if __name__ == "__main__":
    # extract(dim=256, size=2)
    # extract(base="vgg11", dim=64, size=2)
    extract(base="res18", dim=128, size=0)
