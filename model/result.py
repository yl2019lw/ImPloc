#!/usr/bin/env python
# -*- coding: utf-8 -*-

# merge kfold result

import os
import pandas as pd


def merge_agg(fv="slf"):
    if fv == 'slf':
        srcdir = os.path.join("result-paper/agg_slf")
    elif fv == "cnnfeat":
        srcdir = os.path.join("result-paper/cnnfeat")
    else:
        srcdir = os.path.join("result-paper/agg_resnet")

    methods = os.listdir(srcdir)
    folds = list(range(1, 11))
    perf = {}
    for m in methods:
        for fold in folds:
            sid = m + "_" + fv + "_fold" + str(fold)
            perf[sid] = {}
            path = os.path.join(srcdir, m, "fold%d.txt" % fold)
            perf[sid]['model'] = m
            perf[sid]["fold"] = fold
            with open(path, 'r') as f:
                for line in f.readlines():
                    key, value = [
                        x.strip() for x in line.strip("\n").split(":")]
                    perf[sid][key] = value

    df = pd.DataFrame(perf).T
    outf = os.path.join(srcdir, "result.csv")
    df.to_csv(outf)


if __name__ == "__main__":
    # merge_agg("slf")
    # merge_agg("resnet")
    merge_agg("cnnfeat")
