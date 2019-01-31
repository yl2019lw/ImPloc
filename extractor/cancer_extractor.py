#!/usr/bin/env python
# -*- coding: utf-8 -*-

# extract res18-128 fv for cancer image
# save fv for each image separately

import cv2
import gpustat
import numpy as np
import torch
import torch.nn as nn
import torchvision
import os
import time
import queue
import threading
import glob


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

CIMG_DIR = '/ndata/longwei/hpa/cancerdata'
CFV_DIR = '/ndata/longwei/hpa/cancerfv_all'
PROJECT_DIR = '/data/longwei/hpa'


def get_gpu_usage(device=1):
    gpu_stats = gpustat.new_query()
    item = gpu_stats.jsonify()["gpus"][device]
    return item['memory.used'] / item['memory.total']


def get_gene_list():
    genes = []
    genelist = os.path.join(PROJECT_DIR, "spider/genelist", "enhanced.list")
    with open(os.path.abspath(genelist), 'r') as f:
        for line in f.readlines():
            gene = line.strip("\n")
            genes.append(gene)
    return genes


def get_gene_all_pics(gene):
    pattern = "%s/%s/*/*.jpg" % (CIMG_DIR, gene)
    return glob.glob(pattern)


def extract_image_fv(q, model, i):

    def _extract_image(gene, image):

        while get_gpu_usage(1) > 0.9:
            print("---gpu full---", get_gpu_usage(1))
            time.sleep(1)
            torch.cuda.empty_cache()

        name = os.path.basename(image).split(".")[0]
        # if hash(name) % 4 != 3:
        #     return

        gene_dir = os.path.join(CFV_DIR, gene)
        tgt_path = os.path.join(gene_dir, '%s.npy' % name)
        if os.path.exists(tgt_path):
            # print("already extract for %s" % tgt_path)
            return

        print("extract for %s" % image)
        img = cv2.imread(image)
        img = cv2.resize(img, (3000, 3000), interpolation=cv2.INTER_CUBIC)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        inputs = torch.from_numpy(img).type(torch.cuda.FloatTensor)
        pred = model(inputs)
        pd = pred.data.cpu().numpy()
        np.save(tgt_path, pd)

    while True:

        gene = q.get()
        if gene is None:
            break
        print("---extract queue-----", gene, q.qsize(), i)
        gene_dir = os.path.join(CFV_DIR, gene)
        if not os.path.exists(gene_dir):
            os.mkdir(gene_dir)

        for p in get_gene_all_pics(gene):
            _extract_image(gene, p)
            # time.sleep(0.1)

        q.task_done()


class Extractor(nn.Module):

    def __init__(self):
        super(Extractor, self).__init__()
        origin = torchvision.models.resnet18(pretrained=True)
        self.end_layer = -4

        self.features = torch.nn.Sequential(
            *list(origin.children())[:self.end_layer])

        for p in self.features.parameters():
            p.require_grad = False

    def forward(self, x):
        f = self.features(x)
        f = torch.nn.AdaptiveAvgPool2d(1)(f)
        return f.view(f.size(0), -1)


def extract():
    q = queue.Queue()
    for gene in get_gene_list():
        q.put(gene)

    model = Extractor()
    model.share_memory()
    model.cuda()

    jobs = []
    for i in range(8):
        p = threading.Thread(target=extract_image_fv, args=(q, model, i))
        jobs.append(p)
        p.start()

    q.join()

    for i in range(8):
        q.put(None)

    for j in jobs:
        j.join()


if __name__ == "__main__":
    # get_gene_list()
    extract()
