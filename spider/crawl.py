#!/usr/bin/env python
# -*-: coding:utf-8 -*-

# crawl from qualified image url list.
# no need to parse xml filed any more.
# used to crawl supported & approved genes.

import urllib
import urllib.request
import os
import time
import random
import threading
import queue


def fetch_img_by_gene(geneList):
    while not geneList.empty():
        gene = geneList.get()
        imageList = []

        # with open("piclist/supported/%s.piclist" % gene) as f:
        with open("piclist/approved/%s.piclist" % gene) as f:
            imageList = [line.strip("\n") for line in f.readlines()]

        # outdir = "/ndata/longwei/hpa/data/"
        outdir = "/ndata/longwei/hpa/approve_data/"
        imgdir = os.path.join(outdir, gene)
        if not os.path.exists(imgdir):
            os.mkdir(imgdir)

        for imgUrl in imageList:
            imgname = imgUrl.split("/")[-1]
            imgpath = os.path.join(imgdir, imgname)
            if not os.path.exists(imgpath):
                time.sleep(random.random())
                req = urllib.request.Request(imgUrl)
                req.add_header('User-agent', 'Mozilla 5.10')
                response = urllib.request.urlopen(req)
                html = response.read()
                with open(imgpath, "wb") as f:
                    f.write(html)
        print("crawl %s finish" % gene, len(imageList))
        time.sleep(1)


def main():
    # genelist = "genelist/supported.list"
    genelist = "genelist/approved.list"
    geneList = queue.Queue()

    with open(genelist, 'r') as f:
        for line in f.readlines():
            geneList.put(line.strip("\n"))

    nt = 32
    threads = []
    for i in range(nt):
        t = threading.Thread(target=fetch_img_by_gene, args=(geneList,))
        threads.append(t)
        t.daemon = True
        t.start()

    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
