#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import torch.multiprocessing as mp
# try:
#     mp.set_start_method('spawn')
# except Exception:
#     pass
import torch
import torchvision
import finetune
import os
import cv2
import numpy as np
# import time
import gpustat
import threading
import queue

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

PATCH_DIR = "/ndata/longwei/hpa/patch200"
FV_DIR = "/ndata/longwei/hpa/fv/res18"


def get_gpu_usage(device=1):
    gpu_stats = gpustat.new_query()
    item = gpu_stats.jsonify()["gpus"][device]
    return item['memory.used'] / item['memory.total']


def extract_image_fv(q, model):

    def _extract_patch(patch):
        img = cv2.imread(os.path.join(image, patch))
        h, w, c = img.shape
        # There may exists some bug image not delete with patch200.py.
        if h != 200 or w != 200:
            return None
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        return img
        # while get_gpu_usage() > 0.9:
        #     print("---gpu full---", get_gpu_usage(), patch)
        #     time.sleep(3)

        # inputs = torch.from_numpy(img).type(torch.cuda.FloatTensor)
        # pd = model(inputs)
        # return pd

    while not q.empty():
        image = q.get()
        print("---extract image-----", image)
        img_name = image.split("/")[-1]
        gene_name = image.split("/")[-2]
        outdir = os.path.join(FV_DIR, gene_name)
        outpath = os.path.join(outdir, "%s.npy" % img_name)
        if os.path.exists(outpath):
            print("------already extracted---------", image)
            continue
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        pimgs = [_extract_patch(os.path.join(image, p))
                 for p in os.listdir(image)]
        pimgs = [x for x in pimgs if x is not None]
        if pimgs:
            inputs = np.concatenate(pimgs, axis=0)
            tinputs = torch.from_numpy(inputs).type(torch.cuda.FloatTensor)
            value = model(tinputs)
            print("----save-----", outpath)
            np.save(outpath, value)

        # try:
        #     pds = [_extract_patch(os.path.join(image, p))
        #            for p in os.listdir(image)]
        #     value = np.concatenate(pds, axis=0)
        #     print("----save-----", outpath)
        #     np.save(outpath, value)
        # except Exception as e:
        #     print("extract failed for", image)
        #     print(e)


def extract():
    # pytorch only support mp start method in ['forserver', 'spawn']
    # print(mp.get_start_method())
    # q = mp.Queue()
    q = queue.Queue()
    for gene in os.listdir(PATCH_DIR):
        gene_dir = os.path.join(PATCH_DIR, gene)
        for img in os.listdir(gene_dir):
            q.put(os.path.join(gene_dir, img))

    resnet18 = torchvision.models.resnet18(pretrained=True)
    model = finetune.FineTuneModel(resnet18)
    for param in model.parameters():
        param.requires_grad = False
    model.share_memory()
    model.cuda()

    jobs = []
    for i in range(20):
        # p = mp.Process(target=extract_image_fv, args=(q, model))
        p = threading.Thread(target=extract_image_fv, args=(q, model))
        jobs.append(p)
        p.daemon = True
        p.start()

    for j in jobs:
        j.join()


def test_image():
    image = "%s/ENSG00000278845/51551_B_9_8" % PATCH_DIR
    resnet18 = torchvision.models.resnet18(pretrained=True)
    model = finetune.FineTuneModel(resnet18)
    for param in model.parameters():
        param.requires_grad = False
    model.cuda()

    def _extract_patch(patch):
        img = cv2.imread(os.path.join(image, patch))
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        inputs = torch.from_numpy(img).type(torch.cuda.FloatTensor)
        pd = model(inputs)
        return pd

    pds = [_extract_patch(os.path.join(image, p)) for p in os.listdir(image)]
    value = torch.cat(pds, dim=0)

    img_name = image.split("/")[-1]
    gene_name = image.split("/")[-2]
    print(img_name, gene_name)
    print(value.shape)
    np.save("%s.npy" % img_name, value)


def test_patch():
    patch = "%s/ENSG00000278845/51551_B_9_8/51551_B_9_8_p9.jpg" % PATCH_DIR
    resnet18 = torchvision.models.resnet18(pretrained=True)
    model = finetune.FineTuneModel(resnet18)
    for param in model.parameters():
        param.requires_grad = False
    model.cuda()

    img = cv2.imread(patch)
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    inputs = torch.from_numpy(img).type(torch.cuda.FloatTensor)
    pd = model(inputs)
    print(pd)
    print(pd.shape)


if __name__ == "__main__":
    # test_patch()
    # test_image()
    extract()
