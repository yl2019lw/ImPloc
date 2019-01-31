#!/usr/bin/env python
# -*- coding:utf-8 -*-

# parse origin label from subcellular_location.tsv
# generate raw label, need merge label according to cellDict.

import os
import pandas as pd

nrow = 2000


def extract_enhanced(region):
    enhanced = region[region.Reliability.isin(['Enhanced'])]
    enhanced = enhanced.copy()
    enhanced['label'] = enhanced[['Enhanced']].astype(str).sum(axis=1)
    enhanced_data = enhanced.loc[:, ['Gene', 'label']]
    ft = enhanced_data['label'] != ""
    enhanced_data = enhanced_data[ft]
    # enhanced_fname = "enhanced_%d.txt" % region.shape[0]
    enhanced_fname = "enhanced.txt"
    enhanced_data.to_csv(enhanced_fname, index=False)


def extract_supported(region):
    supported = region[region.Reliability.isin(['Enhanced', 'Supported'])]
    supported = supported.copy()
    supported['label'] = supported[['Enhanced', 'Supported']].apply(
                lambda x: ';'.join(x), axis=1)
    supported_data = supported.loc[:, ['Gene', 'label']]
    ft = supported_data['label'] != ""
    supported_data = supported_data[ft]
    # supported_fname = "supported_%d.txt" % region.shape[0]
    supported_fname = "supported.txt"
    supported_data.to_csv(supported_fname, index=False)


def extract_approved(region):
    approved = region[region.Reliability.isin(
                ['Enhanced', 'Supported', 'Approved'])]
    approved = approved.copy()
    approved['label'] = approved[['Enhanced', 'Supported', 'Approved']].apply(
            lambda x: ';'.join(x), axis=1)
    approved_data = approved.loc[:, ['Gene', 'label']]
    ft = approved_data['label'] != ""
    approved_data = approved_data[ft]
    # approved_fname = "approved_%d.txt" % region.shape[0]
    approved_fname = "approved.txt"
    approved_data.to_csv(approved_fname, index=False)


def extract_uncertain(region):
    uncertain = region[region.Reliability.isin(
                ['Enhanced', 'Supported', 'Approved', 'Uncertain'])]
    uncertain = uncertain.copy()
    uncertain['label'] = uncertain[['Enhanced', 'Supported',
                                    'Approved', 'Uncertain']].apply(
                                        lambda x: ';'.join(x), axis=1)
    uncertain_data = uncertain.loc[:, ['Gene', 'label']]
    ft = uncertain_data['label'] != ""
    uncertain_data = uncertain_data[ft]
    uncertain_fname = "uncertain_%d.txt" % region.shape[0]
    uncertain_data.to_csv(uncertain_fname, index=False)


if __name__ == "__main__":
    src = os.path.join(os.path.pardir, "spider/genelist",
                       "subcellular_location.tsv")
    data = pd.read_table(src, na_filter=False)
    # region = data[0:nrow]
    region = data
    # extract_enhanced(region)
    # extract_supported(region)
    extract_approved(region)
    # extract_uncertain(region)
