#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torchvision
import finetune
import os
import cv2
import numpy as np
import time
import gpustat
import threading
import queue
import glob

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# for tissue data
DATA_DIR = "/data/longwei/hpa/qdata"
SUPP_DATA_DIR = "/ndata/longwei/hpa/data"

TISSUE_DIR = "/ndata/longwei/hpa/tissuedata"
FV_DIR = "/ndata/longwei/hpa/tissuefv/res18"


def get_gpu_usage(device=1):
    gpu_stats = gpustat.new_query()
    item = gpu_stats.jsonify()["gpus"][device]
    return item['memory.used'] / item['memory.total']


def get_gene_list():
    pattern = "%s/**/*.txt" % TISSUE_DIR
    genes = [os.path.splitext(os.path.basename(x))[0]
             for x in glob.glob(pattern)]
    return list(set(genes))


def get_gene_pics(gene):
    pics = []
    for t in ['liver', 'breast', 'prostate', 'bladder']:
        tp = os.path.join(TISSUE_DIR, t, "%s.txt" % gene)
        if os.path.exists(tp):
            with open(tp, 'r') as f:
                pics.extend([l.strip("\n") for l in f.readlines()])
    return pics


def extract_image_fv(q, model, i):

    def _extract_image(image):

        img = cv2.imread(image)
        img = cv2.resize(img, (3000, 3000), interpolation=cv2.INTER_CUBIC)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        # return img

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

        outpath = os.path.join(FV_DIR, "%s.npy" % gene)
        if os.path.exists(outpath):
            print("------already extracted---------", gene)
            q.task_done()
            continue

        pds = [_extract_image(os.path.join(gene_dir, p))
               for p in get_gene_pics(gene)
               if os.path.splitext(p)[-1] == ".jpg"]
        if pds:
            value = np.concatenate(pds, axis=0)
            print("----save-----", outpath)
            np.save(outpath, value)
        q.task_done()


def extract():
    q = queue.Queue()
    for gene in get_gene_list():
        q.put(gene)

    resnet18 = torchvision.models.resnet18(pretrained=True)
    model = finetune.FineTuneModel(resnet18)
    for param in model.parameters():
        param.requires_grad = False
    model.share_memory()
    model.cuda()
    # print(model)

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
