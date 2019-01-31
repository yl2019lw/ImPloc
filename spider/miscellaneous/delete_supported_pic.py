#!/usr/bin/env python
# -*- coding: utf-8 -*-

# delete original crawled supported image whose staining quality is low

import os
import multiprocessing as mp
import time
import random

outdir = "/ndata/longwei/hpa/data"


def handle_gene(q):
    while True:
        time.sleep(random.random())
        if q.empty():
            break
        gene = q.get()
        genedir = os.path.join(outdir, gene)
        allfiles = [os.path.join(genedir, f) for f in os.listdir(genedir)]

        piclist = os.path.join(os.path.pardir,
                               'piclist/supported', "%s.piclist" % gene)
        qfiles = []
        with open(piclist, 'r') as f:
            picList = [line.strip("\n").split("/")[-1]
                       for line in f.readlines()]
            qfiles = [os.path.join(genedir, pic) for pic in picList]

        for path in allfiles:
            if path in qfiles:
                pass
            else:
                os.system("rm -rf %s" % path)
        print("handel gene %s finish" % gene)


def main():
    genelist = os.path.join(os.path.pardir, "genelist/supported_2000.list")
    geneList = []
    with open(genelist, 'r') as f:
        geneList = [line.strip("\n") for line in f.readlines()]

    q = mp.Queue()
    for gene in geneList:
        q.put(gene)

    jobs = []
    for i in range(20):
        p = mp.Process(target=handle_gene, args=(q,))
        jobs.append(p)
        p.daemon = True
        p.start()

    for j in jobs:
        j.join()


if __name__ == "__main__":
    main()
