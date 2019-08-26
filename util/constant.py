#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

PROJECT = os.path.join(os.path.dirname(__file__), os.pardir)
ROOT = os.path.join(PROJECT, "data")

# enhanced img dir
DATA_DIR = os.path.join(ROOT, "enhanced_4tissue_imgs")

# enhanced qdata img dir
QDATA_DIR = os.path.join(ROOT, "enhanced_4tissue_imgs")

# supported img dir
SUPP_DATA_DIR = os.path.join(ROOT, "supported_imgs")

# approved img dir
APPROVE_DATA_DIR = os.path.join(ROOT, "approved_4tissue_imgs")

# enhanced 4tissue pic list
TISSUE_DIR = os.path.join(ROOT, "enhanced_4tissue_piclist")

# enhanced all tissue pic list
ALL_TISSUE_DIR = os.path.join(ROOT, "enhanced_all_piclist")

# enhanced 4tissue fv
FV_DIR = os.path.join(ROOT, "enhanced_4tissue_fv", "res18_128")

# enhanced all tissue fv
ALL_FV_DIR = os.path.join(ROOT, "enhanced_all_fv", "res18_128")

# iLocator fv
MATLAB_FV_DIR = os.path.join(ROOT, "ilocator_fv")

# cancer fv pic list
CANCER_DATA_DIR = os.path.join(ROOT, "enhanced_cancerfv_3tissue_piclist")

# cancer fv
CANCER_FV_DIR = os.path.join(ROOT, "enhanced_cancerfv_3tissue")


if __name__ == "__main__":
    pass
