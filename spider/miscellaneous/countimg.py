#!/usr/bin/env python
# -*- coding:utf-8 -*-

from bs4 import BeautifulSoup


def countimg():
    count = {}
    with open("genelist.txt") as f:
        for line in f.readlines():
            gene = line.strip("\n")
            count[gene] = 0
            with open("xml/%s.xml" % gene, 'r') as xf:
                soup = BeautifulSoup(xf, "xml")
                tissueExpression_all = soup.find_all("tissueExpression")
                for tE in tissueExpression_all:
                    imageUrl_all = tE.find_all("imageUrl")
                    count[gene] += len(imageUrl_all)
    return count


if __name__ == "__main__":
    cnt = countimg()
    total = 0
    f = open("count", "w")
    for gene, c in cnt.items():
        f.write(str(gene) + "," + str(c))
        f.write("\n")
        total += c
    f.close()
    print("total img count is:", total)
