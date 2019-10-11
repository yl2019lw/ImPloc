#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from model import mtrain
from model import img_rnn
from util import torch_util

# matlab fv use batchsize=64
# batchsize = 256
batchsize = 64


def transformer_bce(fv, size=0):
    model_name = "transformer_%s_size%d_bce" % (fv, size)
    criterion = torch.nn.BCELoss(reduction='none')
    mtrain.train(fv, model_name, criterion, batchsize=batchsize)


def transformer_fec1(fv, size=0):
    model_name = "transformer_%s_size%d_fec1" % (fv, size)
    criterion = torch_util.FECLoss(alpha=batchsize*1, reduction='none')
    mtrain.train(fv, model_name, criterion)


def transformer_bce_gbalance(fv, size=0):
    model_name = "transformer_%s_size%d_bce_gbalance" % (fv, size)
    criterion = torch.nn.BCELoss(reduction='none')
    mtrain.train(fv, model_name, criterion, balance=True, batchsize=batchsize)


def transformer_bce_kfold(fv):
    folds = list(range(1, 11))
    for fold in folds:
        model_name = "transformer_%s_bce_fold%d" % (fv, fold)
        criterion = torch.nn.BCELoss(reduction='none')
        mtrain.train(fv, model_name, criterion, balance=False,
                     batchsize=batchsize, fold=fold)


def fix_break_run():
    folds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fv = 'matlab'
    for fold in folds:
        model_name = "transformer_%s_bce_fold%d" % (fv, fold)
        criterion = torch.nn.BCELoss(reduction='none')
        mtrain.train(fv, model_name, criterion, balance=False,
                     batchsize=32, fold=fold)


def imgrnn_bce_kfold():
    folds = list(range(1, 11))
    # fvs = ["matlab", "res18-128"]
    fvs = ['res18-128']
    for fv in fvs:
        for fold in folds:
            img_rnn.train(fv, fold=fold)


if __name__ == "__main__":
    # transformer_bce("res18-64")
    # transformer_bce("res18-128")
    # transformer_bce("res18-256")
    # transformer_bce("res18-512")
    # transformer_bce("matlab")
    # transformer_bce_gbalance("matlab")
    # transformer_fec1("res18-64")
    # transformer_fec1("res18-128")
    # transformer_fec1("res18-256")
    # transformer_fec1("res18-512")
    # transformer_fec5("matlab")
    # transformer_bce_kfold("res18-512")
    # fix_break_run()
    imgrnn_bce_kfold()
