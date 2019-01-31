#!/usr/bin/env python
# -*- coding: utf-8 -*-

# parse qualified image list by staining level from xml files
# add parse_tissue_xml to reduce the amount of images for approved genes.

import os
from bs4 import BeautifulSoup
import multiprocessing as mp

tissue_list = ['liver', 'breast', 'prostate', 'bladder']


def parse_xml(xml_list):
    '''parse qualified image list by staining level from xml files'''
    while not xml_list.empty():
        xml = xml_list.get()
        gene = os.path.splitext(os.path.basename(xml))[0]
        outf = os.path.join(os.path.pardir, "piclist", "%s.piclist" % gene)
        picList = []
        with open(xml, 'r') as f:
            soup = BeautifulSoup(f.read(), 'xml')
            te_list = soup.find_all('tissueExpression', {"technology": "IHC"})
            for te in te_list:
                data_list = te.find_all("data")
                for data in data_list:
                    level = data.tissueCell.find("level",
                                                 {"type": "intensity"})
                    lflag = level and level.get_text() in ['Strong',
                                                           'Moderate']
                    quantity = data.tissueCell.find("quantity")
                    qflag = quantity and quantity.get_text() in ['>75%']
                    if lflag and qflag:
                        for imageUrl in data.find_all("imageUrl"):
                            picList.append(imageUrl.get_text())
        with open(os.path.abspath(outf), 'w') as f:
            for pic in picList:
                f.write("%s\n" % pic)


def parse_tissue_xml(xml_list):
    '''parse qualified image for 4 tissues(liver...)'''
    while not xml_list.empty():
        xml = xml_list.get()
        gene = os.path.splitext(os.path.basename(xml))[0]
        outf = os.path.join(os.path.pardir, "piclist", "%s.piclist" % gene)
        picList = []
        with open(xml, 'r') as f:
            soup = BeautifulSoup(f.read(), 'xml')
            te_list = soup.find_all('tissueExpression', {"technology": "IHC"})
            for te in te_list:
                data_list = te.find_all("data")
                for data in data_list:
                    # add tissue limit here
                    t = data.tissue.get_text()
                    if t not in ['liver', 'breast',
                                 'prostate', 'urinary bladder']:
                        continue
                    level = data.tissueCell.find("level",
                                                 {"type": "intensity"})
                    lflag = level and level.get_text() in ['Strong',
                                                           'Moderate']
                    quantity = data.tissueCell.find("quantity")
                    qflag = quantity and quantity.get_text() in ['>75%']
                    if lflag and qflag:
                        for imageUrl in data.find_all("imageUrl"):
                            picList.append(imageUrl.get_text())
        with open(os.path.abspath(outf), 'w') as f:
            for pic in picList:
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
        # p = mp.Process(target=parse_xml, args=(xml_list,))
        p = mp.Process(target=parse_tissue_xml, args=(xml_list,))
        p.daemon = True
        jobs.append(p)
        p.start()

    for j in jobs:
        j.join()


if __name__ == "__main__":
    # genelist = os.path.join(os.path.pardir, "genelist", "enhanced.list")
    # genelist = os.path.join(os.path.pardir, "genelist", "supported.list")
    genelist = os.path.join(os.path.pardir, "genelist", "approved.list")
    parse_list(genelist)
