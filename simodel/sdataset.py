#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import json
from torch.utils.data import Dataset
from util import datautil
from util import constant as c

DATA_DIR = c.DATA_DIR
NUM_CLASSES = 6


class SImgDataset(Dataset):

    def __init__(self, mode="train", size=3000):
        super(SImgDataset, self).__init__()
        self.size = size
        self.gene_label = datautil.load_gene_label(0)
        all_genes = datautil.get_gene_list(0)
        all_genes = [gene for gene in all_genes if gene in self.gene_label]

        spivot = int(len(all_genes) * 0.7)
        epivot = int(len(all_genes) * 0.9)
        if mode == "train":
            self.genes = all_genes[:spivot]
        elif mode == "val":
            self.genes = all_genes[spivot:epivot]
        elif mode == "test":
            self.genes = all_genes[epivot:]
        else:
            raise Exception("Unknown mode in SImgDataset", mode)

        self.gene_imgs = [(gene, img) for gene in self.genes
                          for img in datautil.get_gene_pics(gene)]

    def __len__(self):
        return len(self.gene_imgs)

    def __getitem__(self, idx):
        gene, img = self.gene_imgs[idx]
        label = self.gene_label[gene]
        imgpath = os.path.join(DATA_DIR, gene, img)
        nimg = cv2.imread(imgpath)
        nimg = cv2.resize(nimg, (self.size, self.size),
                          interpolation=cv2.INTER_CUBIC)
        nimg = np.transpose(nimg, (2, 0, 1))
        nlabel = np.zeros(NUM_CLASSES)
        for l in label:
            nlabel[l] = 1

        return (nimg, nlabel)


def choice(seq, n=20):
    '''random chosse n elements from seq'''
    if len(seq) > n:
        idx = np.random.randint(len(seq), size=n)
        return [seq[x] for x in idx]
    else:
        return seq


def mid_choice(seq, n=50):
    '''choose middle n elements from seq'''
    e = len(seq)
    if e < n:
        return []
    else:
        s = (e - n) // 2
        return seq[s:s+n]


class SPatchDataset(Dataset):
    '''choose middle npatch from single image, read image only once'''

    def __init__(self, mode="train", roiSize=100, npatch=32):
        super(SPatchDataset, self).__init__()
        self.roiSize = roiSize
        self.npatch = npatch
        self.gene_label = datautil.load_gene_label(0)
        all_genes = datautil.get_gene_list(0)
        all_genes = [gene for gene in all_genes if gene in self.gene_label]

        spivot = int(len(all_genes) * 0.7)
        epivot = int(len(all_genes) * 0.9)
        if mode == "train":
            self.genes = all_genes[:spivot]
        elif mode == "val":
            self.genes = all_genes[spivot:epivot]
        elif mode == "test":
            self.genes = all_genes[epivot:]
        else:
            raise Exception("Unknown mode in SPatchDataset", mode)

        with open("roi%d.json" % self.roiSize, 'r') as f:
            self.roi = json.load(f)

        self.gene_imgs = [(gene, img) for gene in self.genes
                          for img in datautil.get_gene_pics(gene)
                          if self.valid_img((gene, img))]
        # self.gene_img_points = [(gene, img, point) for gene in self.genes
        #                         for img in self.roi[gene].keys()
        #                         for point in mid_choice(self.roi[gene][img])]

    def valid_img(self, item):
        gene, img = item
        points = self.roi[gene][img]
        if len(points) < self.npatch:
            return False
        return True

    def __len__(self):
        return len(self.gene_imgs)

    def __getitem__(self, idx):
        gene, img = self.gene_imgs[idx]
        points = self.roi[gene][img]
        label = self.gene_label[gene]
        imgpath = os.path.join(DATA_DIR, gene, img)
        nimg = cv2.imread(imgpath)
        nimg = np.transpose(nimg, (2, 0, 1))
        nlabel = np.zeros(NUM_CLASSES)
        for l in label:
            nlabel[l] = 1

        patches = []
        labels = []
        points = mid_choice(points, self.npatch)
        for point in points:
            xc, yc = [int(x) for x in point]
            xs = xc - self.roiSize // 2
            xe = xc + self.roiSize // 2
            ys = yc - self.roiSize // 2
            ye = yc + self.roiSize // 2
            patch = nimg[:, xs:xe, ys:ye]
            patches.append(patch)
            labels.append(nlabel)

        return np.stack(np.array(patches)), np.stack(np.array(labels))


class SPatchAllDataset(Dataset):
    '''track idx, return all patches of a single image'''

    def __init__(self, mode="train", roiSize=100):
        super(SPatchAllDataset, self).__init__()
        self.roiSize = roiSize
        self.gene_label = datautil.load_gene_label(0)
        all_genes = datautil.get_gene_list(0)
        all_genes = [gene for gene in all_genes if gene in self.gene_label]

        spivot = int(len(all_genes) * 0.7)
        epivot = int(len(all_genes) * 0.9)
        if mode == "train":
            self.genes = all_genes[:spivot]
        elif mode == "val":
            self.genes = all_genes[spivot:epivot]
        elif mode == "test":
            self.genes = all_genes[epivot:]
        else:
            raise Exception("Unknown mode in SPatchAllDataset", mode)

        with open("roi%d.json" % self.roiSize, 'r') as f:
            self.roi = json.load(f)

        self.gene_img_points = [(gene, img, point) for gene in self.genes
                                for img in self.roi[gene].keys()
                                for point in self.roi[gene][img]]

        self.cache_img = None
        self.cache_nimg = None

    def __len__(self):
        return len(self.gene_img_points)

    def __getitem__(self, idx):
        gene, img, point = self.gene_img_points[idx]
        xc, yc = [int(x) for x in point]
        label = self.gene_label[gene]

        if self.cache_img == img:
            nimg = self.cache_nimg
        else:
            self.cache_img = img
            imgpath = os.path.join(DATA_DIR, gene, img)
            nimg = cv2.imread(imgpath)
            nimg = np.transpose(nimg, (2, 0, 1))
            self.cache_nimg = nimg

        xc, yc = [int(x) for x in point]
        xs = xc - self.roiSize // 2
        xe = xc + self.roiSize // 2
        ys = yc - self.roiSize // 2
        ye = yc + self.roiSize // 2

        patch = nimg[:, xs:xe, ys:ye]

        nlabel = np.zeros(NUM_CLASSES)
        for l in label:
            nlabel[l] = 1

        return (patch, nlabel)


class ShufflePatchDataset(Dataset):
    '''shuffle all patches, much slower'''

    def __init__(self, mode='train', roiSize=100):
        super(ShufflePatchDataset).__init__()
        self.roiSize = roiSize
        self.gene_label = datautil.load_gene_label(0)
        all_genes = datautil.get_gene_list(0)
        all_genes = [gene for gene in all_genes if gene in self.gene_label]

        spivot = int(len(all_genes) * 0.7)
        epivot = int(len(all_genes) * 0.9)
        if mode == "train":
            self.genes = all_genes[:spivot]
        elif mode == "val":
            self.genes = all_genes[spivot:epivot]
        elif mode == "test":
            self.genes = all_genes[epivot:]
        else:
            raise Exception("Unknown mode in SPatchAllDataset", mode)

        with open("roi%d.json" % self.roiSize, 'r') as f:
            self.roi = json.load(f)

        self.gene_img_points = [(gene, img, point) for gene in self.genes
                                for img in self.roi[gene].keys()
                                for point in self.roi[gene][img]]
        if mode == "train":
            idx = np.random.permutation(range(len(self.gene_img_points)))
            self.gene_img_points = [self.gene_img_points[i] for i in idx]

    def __len__(self):
        return len(self.gene_img_points)

    def __getitem__(self, idx):
        gene, img, point = self.gene_img_points[idx]
        xc, yc = [int(x) for x in point]
        label = self.gene_label[gene]
        xs = xc - self.roiSize // 2
        xe = xc + self.roiSize // 2
        ys = yc - self.roiSize // 2
        ye = yc + self.roiSize // 2

        imgpath = os.path.join(DATA_DIR, gene, img)
        nimg = cv2.imread(imgpath)
        nimg = np.transpose(nimg, (2, 0, 1))
        patch = nimg[:, xs:xe, ys:ye]

        nlabel = np.zeros(NUM_CLASSES)
        for l in label:
            nlabel[l] = 1

        return (patch, nlabel)


class DecodePatchDataset(Dataset):
    '''load already decoded patch'''
    def __init__(self, mode='train', roiSize=224):
        super(DecodePatchDataset).__init__()
        self.roiSize = roiSize
        self.patch_dir = os.path.expanduser("~/patch%s" % self.roiSize)
        self.gene_label = datautil.load_gene_label(0)
        all_genes = datautil.get_gene_list(0)
        all_genes = [gene for gene in all_genes if gene in self.gene_label]

        spivot = int(len(all_genes) * 0.7)
        epivot = int(len(all_genes) * 0.9)
        if mode == "train":
            self.genes = all_genes[:spivot]
        elif mode == "val":
            self.genes = all_genes[spivot:epivot]
        elif mode == "test":
            self.genes = all_genes[epivot:]
        else:
            raise Exception("Unknown mode in SPatchAllDataset", mode)

        with open("roi%d.json" % self.roiSize, 'r') as f:
            self.roi = json.load(f)

        self.gene_img_pids = [(gene, img, pid) for gene in self.genes
                              for img in self.roi[gene].keys()
                              for pid in range(len(self.roi[gene][img]))]
        if mode == "train":
            idx = np.random.permutation(range(len(self.gene_img_pids)))
            self.gene_img_pids = [self.gene_img_pids[i] for i in idx]

    def __len__(self):
        return len(self.gene_img_pids)

    def __getitem__(self, idx):
        gene, img, pid = self.gene_img_pids[idx]
        gene_dir = os.path.join(self.patch_dir, gene)
        imgpt = os.path.join(gene_dir, "%s.%d.npy" % (img, pid))
        patch = np.load(imgpt)

        label = self.gene_label[gene]
        nlabel = np.zeros(NUM_CLASSES)
        for l in label:
            nlabel[l] = 1

        return (patch, nlabel)


class SmallPatchDataset(Dataset):
    '''load raw small patch in a separate file'''
    def __init__(self, mode='train', roiSize=224,
                 balance=False, transform=None):
        super(SmallPatchDataset).__init__()
        self.roiSize = roiSize
        self.transform = transform
        self.patch_dir = os.path.expanduser("~/patch%s" % self.roiSize)
        self.gene_label = datautil.load_gene_label(0)

        all_genes = datautil.get_gene_list(0)
        all_genes = [gene for gene in all_genes if gene in self.gene_label]

        spivot = int(len(all_genes) * 0.7)
        epivot = int(len(all_genes) * 0.9)
        if mode == "train":
            self.genes = all_genes[:spivot]
            if balance:
                self.genes = datautil.get_balanced_gene_list(self.genes, 0)
        elif mode == "val":
            self.genes = all_genes[spivot:epivot]
        elif mode == "test":
            self.genes = all_genes[epivot:]
        else:
            raise Exception("Unknown mode in SPatchAllDataset", mode)

        with open("roi/roi%d.json" % self.roiSize, 'r') as f:
            self.roi = json.load(f)

        self.gene_img_pids = [(gene, img, pid) for gene in self.genes
                              for img in self.roi[gene].keys()
                              for pid in range(len(self.roi[gene][img]))]
        if mode == "train":
            idx = np.random.permutation(range(len(self.gene_img_pids)))
            self.gene_img_pids = [self.gene_img_pids[i] for i in idx]

    def __len__(self):
        return len(self.gene_img_pids)

    def __getitem__(self, idx):
        gene, img, pid = self.gene_img_pids[idx]
        gene_dir = os.path.join(self.patch_dir, gene)
        imgpt = os.path.join(gene_dir, "%s.%d.jpg" % (img, pid))

        nimg = cv2.imread(imgpt)
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        nimg = np.transpose(nimg, (2, 0, 1))
        patch = nimg

        if self.transform:
            patch = self.transform(patch)
        # patch = patch.astype('float')

        label = self.gene_label[gene]
        nlabel = np.zeros(NUM_CLASSES)
        for l in label:
            nlabel[l] = 1

        return (patch, nlabel)


def test_speed():
    import time
    from torch.utils.data import DataLoader
    start = time.time()
    train_dataset = SmallPatchDataset("train", roiSize=224)
    train_loader = DataLoader(train_dataset, batch_size=128,
                              shuffle=True, num_workers=30)
    for i_batch, sample_batched in enumerate(train_loader):
        (img, label) = sample_batched
    end = time.time()
    print("elapsed %f seconds" % (end - start))


if __name__ == "__main__":
    test_speed()
