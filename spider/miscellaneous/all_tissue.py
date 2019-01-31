#!/usr/bin/env python
# -*- coding: utf-8 -*-

# copy from tissue.py, but for all tissues not only 4

import os
from bs4 import BeautifulSoup
import multiprocessing as mp

OUT_DIR = "/ndata/longwei/hpa/all_tissuedata/"


def parse_xml(xml_list):

    while not xml_list.empty():
        xml = xml_list.get()
        gene = os.path.splitext(os.path.basename(xml))[0]
        picList = {}

        with open(xml, 'r') as f:
            soup = BeautifulSoup(f.read(), 'xml')
            te_list = soup.find_all('tissueExpression', {"technology": "IHC"})
            for te in te_list:
                data_list = te.find_all("data")
                for data in data_list:
                    t = data.tissue.get_text()
                    picList[t] = []

                    level = data.tissueCell.find(
                        "level", {"type": "intensity"})
                    lflag = level and level.get_text() in [
                        'Strong', 'Moderate']
                    quantity = data.tissueCell.find("quantity")
                    qflag = quantity and quantity.get_text() in ['>75%']

                    if lflag and qflag:
                        for imageUrl in data.find_all("imageUrl"):
                            picList[t].append(
                                imageUrl.get_text().split("/")[-1])

        for t in picList.keys():
            if not picList[t]:
                continue

            t_dir = os.path.join(OUT_DIR, t)
            if not os.path.exists(t_dir):
                os.mkdir(t_dir)
            outf = os.path.join(OUT_DIR, t, "%s.txt" % gene)
            with open(os.path.abspath(outf), 'w') as f:
                for pic in picList[t]:
                    f.write("%s\n" % pic)


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
    parse_list(genelist)
