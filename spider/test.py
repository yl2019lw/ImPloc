#!/usr/bin/env python
# -*-: coding: utf-8 -*-

# test for crwal data from proteinatlas

import urllib
import urllib.request
import time
import random
import os
from bs4 import BeautifulSoup


def fetch_xml():
    baseurl = "https://www.proteinatlas.org/search"
    gene = "ENSG00000059378"
    url = "%s/%s?format=xml" % (baseurl, gene)
    filename = "%s.xml.gz" % gene
    # print("url", url)

    time.sleep(random.random())
    req = urllib.request.Request(url)
    req.add_header('User-agent', 'Mozilla 5.10')
    response = urllib.request.urlopen(req)
    html = response.read()
    print("html", html)

    with open("xml/%s" % filename, "wb") as f:
        f.write(html)


def fetch_img(gene):
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


if __name__ == "__main__":
    # gene = "ENSG00000059378"
    # fetch_img(gene)
    with open("loss", 'r') as f:
        for line in f.readlines():
            gene = line.strip("\n")
            fetch_img(gene)
