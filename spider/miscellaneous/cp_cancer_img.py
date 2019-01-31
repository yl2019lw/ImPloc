#!/usr/bin/env python
# -*- coding: utf-8 -*-

# copy qualified cancer img from previous download to separate folder

import os
from bs4 import BeautifulSoup
import multiprocessing as mp
import shutil
import urllib
import urllib.request

SRC_DIR = "/data/longwei/hpa/qdata/"
OUT_DIR = "/ndata/longwei/hpa/cancerdata/"


def crawlimg(imgUrl, imgpath):
    # import time
    # import random
    # time.sleep(random.random())
    req = urllib.request.Request(imgUrl)
    req.add_header('User-agent', 'Mozilla 5.10')
    response = urllib.request.urlopen(req)
    html = response.read()
    with open(imgpath, "wb") as f:
        f.write(html)


def parse_xml(xml_list):

    while not xml_list.empty():
        xml = xml_list.get()
        gene = os.path.splitext(os.path.basename(xml))[0]

        src_gene_dir = os.path.join(SRC_DIR, gene)
        gene_dir = os.path.join(OUT_DIR, gene)
        if not os.path.exists(gene_dir):
            os.mkdir(gene_dir)

        picList = {}
        with open(xml, 'r') as f:
            soup = BeautifulSoup(f.read(), 'xml')
            cancer_te_list = soup.find_all(
                'tissueExpression',
                {"technology": "IHC", "assayType": "pathology"})

            for te in cancer_te_list:
                data_list = te.find_all("data")
                for data in data_list:
                    t = data.tissue.get_text()
                    picList[t] = []
                    # print("find tissue:%s for gene:%s" % (t, gene))
                    patients = data.find_all("patient")
                    for pa in patients:
                        level = pa.find(
                            "level", {"type": "intensity"})
                        lflag = level and level.get_text() in [
                            'Strong', 'Moderate']
                        quantity = pa.find("quantity")
                        qflag = quantity and quantity.get_text() in ['>75%']

                        if lflag and qflag:
                            for imageUrl in pa.find_all("imageUrl"):
                                picList[t].append(imageUrl.get_text())

        for t, pics in picList.items():
            if len(pics):
                tissue_dir = os.path.join(gene_dir, t)
                if not os.path.exists(tissue_dir):
                    os.mkdir(tissue_dir)
                for url in pics:
                    p = url.split("/")[-1]
                    src_path = os.path.join(src_gene_dir, p)
                    tgt_path = os.path.join(tissue_dir, p)

                    if os.path.exists(tgt_path):
                        continue

                    if not os.path.exists(src_path):
                        print("%s for %s not exists" % (src_path, gene))
                        crawlimg(url, tgt_path)
                    else:
                        print("copy %s" % tgt_path)
                        shutil.copy(src_path, tgt_path)


def parse_list(genelist):
    xml_list = mp.Queue()
    xml_dir = os.path.abspath(os.path.join(os.path.pardir, "xml"))
    with open(os.path.abspath(genelist), 'r') as f:
        for line in f.readlines():
            gene = line.strip("\n")
            # parse_xml(os.path.join(xml_dir, "%s.xml" % gene))
            xml_list.put(os.path.join(xml_dir, "%s.xml" % gene))

    nt = 20
    jobs = []
    for i in range(nt):
        p = mp.Process(target=parse_xml, args=(xml_list,))
        p.daemon = True
        jobs.append(p)
        p.start()

    for j in jobs:
        j.join()


if __name__ == "__main__":
    genelist = os.path.join(os.path.pardir, "genelist", "enhanced.list")
    # genelist = os.path.join(os.path.pardir, "genelist", "supported.list")
    parse_list(genelist)
