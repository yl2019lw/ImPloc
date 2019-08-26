#!/usr/bin/env python
# -*- coding: utf-8 -*-

# merge fv for individual image for 4 tissue


import numpy as np
import queue
import threading
import os
from bs4 import BeautifulSoup
from util import contant as c


PROJECT_DIR = c.PROJECT
CIMG_DIR = c.CANCER_IMG_DIR
CFV_DIR = c.CANCER_ALL_FV_DIR
CFV_4TISSUE_DIR = c.CANCER_FV_DIR

tissue_list = ['liver cancer', 'breast cancer', 'prostate cancer']


def get_gene_list():
    genes = []
    genelist = os.path.join(PROJECT_DIR, "spider/genelist", "enhanced.list")
    with open(os.path.abspath(genelist), 'r') as f:
        for line in f.readlines():
            gene = line.strip("\n")
            genes.append(gene)
    return genes


def get_cancer_4t_pics(gene):
    # need update to read from cancer_3tissue_list
    picList = []
    xml = os.path.join(PROJECT_DIR, "spider/xml/%s.xml" % gene)
    with open(xml, 'r') as f:
        soup = BeautifulSoup(f.read(), 'xml')
        cancer_te_list = soup.find_all(
            'tissueExpression',
            {"technology": "IHC", "assayType": "pathology"})

        for te in cancer_te_list:
            data_list = te.find_all("data")
            for data in data_list:
                t = data.tissue.get_text()
                if t not in tissue_list:
                    continue

                # print("find tissue:%s for gene:%s" % (t, gene))
                patients = data.find_all("patient")
                for pa in patients:
                    level = pa.find(
                        "level", {"type": "intensity"})
                    lflag = level and level.get_text() in [
                        'Strong', 'Moderate']
                    quantity = pa.find("quantity")
                    qflag = quantity and quantity.get_text() in ['>75%']

                    if not lflag or not qflag:
                        continue

                    for imageUrl in pa.find_all("imageUrl"):
                        picList.append(imageUrl.get_text().split("/")[-1])
    return picList


def merge_gene(q):
    while True:
        gene = q.get()
        if gene is None:
            break
        gene_dir = os.path.join(CFV_DIR, gene)
        gene_fv = []

        for p in get_cancer_4t_pics(gene):
            fvp = os.path.join(gene_dir, p.replace('jpg', 'npy'))
            if not os.path.exists(fvp):
                print("----fv %s not exists" % fvp)
                continue

            gene_fv.append(np.load(fvp))

        if not len(gene_fv):
            print("----gene fv blank---", gene)
            q.task_done()
            continue

        np_fv = np.concatenate(gene_fv)
        tgt_path = os.path.join(CFV_4TISSUE_DIR, "%s.npy" % gene)
        np.save(tgt_path, np_fv)
        q.task_done()


def merge():
    q = queue.Queue()
    for gene in get_gene_list():
        q.put(gene)

    jobs = []
    for i in range(20):
        p = threading.Thread(target=merge_gene, args=(q,))
        jobs.append(p)
        p.start()

    q.join()

    for i in range(20):
        q.put(None)

    for j in jobs:
        j.join()


if __name__ == "__main__":
    merge()
