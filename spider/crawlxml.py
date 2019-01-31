#!/usr/bin/env python
# -*-: coding: utf-8 -*-

# crwal xml files for each gene

import urllib
import urllib.request
import time
import random
from multiprocessing import Pool
from multiprocessing import Queue

q = Queue()
baseurl = "https://www.proteinatlas.org/search"


def fetch_xml_by_gene(i):
    while not q.empty():
        gene = q.get(True)
        # eg. https://www.proteinatlas.org/search/ENSG00000059378?format=xml
        url = "%s/%s?format=xml" % (baseurl, gene)
        filename = "%s.xml.gz" % gene
        # print("url", url)

        time.sleep(random.random())
        req = urllib.request.Request(url)
        req.add_header('User-agent', 'Mozilla 5.10')
        response = urllib.request.urlopen(req)
        html = response.read()
        # print("html", html)

        with open("xml/%s" % filename, "wb") as f:
            f.write(html)


def main():
    with open("genelist/genelist.txt") as f:
        for line in f.readlines():
            q.put(line.strip("\n"))

    print("q size:", q.qsize())
    p = Pool(1000)
    for i in range(q.qsize()):
        p.apply_async(fetch_xml_by_gene, args=(i,))
    p.close()
    p.join()


if __name__ == "__main__":
    main()
