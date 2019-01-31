#!/usr/bin/env python
# -*-: coding:utf-8 -*-

# deprecated, now directly crawl from qualified image url list
# please see crawl.py

import urllib
import urllib.request
import os
import time
import random
from bs4 import BeautifulSoup
import threading
import queue


def fetch_img_by_gene(geneList):
    while True:
        gene = geneList.get()
        imageList = []
        with open("xml/%s.xml" % gene, 'r') as f:
            soup = BeautifulSoup(f, "xml")
            tissueExpression_all = soup.find_all("tissueExpression")
            for tE in tissueExpression_all:
                imageUrl_all = tE.find_all("imageUrl")
                for iU in imageUrl_all:
                    imageList.append(iU.text)

        imgdir = os.path.join(os.path.pardir, "data", gene)
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
        print("crawl %s finish" % gene)
        time.sleep(1)


def main():
    geneList = queue.Queue()

    with open("genelist/genelist.txt") as f:
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
