#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from model import mtrain
from util import torch_util

# imbtrain_data = fvloader.load_train_data(size=size, balance=False)

# train_data = garbage_shuffle(train_data)
# val_data = garbage_shuffle(val_data)
# model_name = "garbage-tv_transformer_%s_size%d_bce_gbalance" % (fv, size)
# model_name = "transformer-h4l3_%s_size%d_bce_gbalance" % (fv, size)
# model_name = "ploss/transformer_%s_size%d_autoloss_alpha64" % (fv, size)
# model_name = "transformer_%s_size%d_sign_balance_decay0.002" % (fv, size)
# model_name = "transformer_%s_size%d_f1loss_balance" % (fv, size)
# model_name = "transformer_%s_size%d_recallloss_balance" % (fv, size)
# model_name = "transformer_%s_size%d_fbetaloss_balance" % (fv, size)

# criterion = torch_util.FbetaLoss()
# criterion = torch_util.RecallLoss()
# criterion = torch_util.F1Loss()
# criterion = torch_util.SignLoss()
# criterion = torch_util.AutoLoss(alpha=64, reduction='none')
# criterion = torch_util.SFocalLoss(reduction='none')
# criterion = torch.nn.BCELoss(reduction='none')
# criterion = torch.nn.BCELoss(reduce=True, size_average=True)
# from util import datautil
# freq = datautil.get_label_freq(size)
# criterion = torch_util.FocalLoss(freq=freq, gamma=2)
# criterion = torch_util.MetricsLoss()

# batchsize 256 for cnn feature vector
batchsize = 256
# batchsize = 64


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


if __name__ == "__main__":
    # transformer_bce("res18-64")
    # transformer_bce("res18-128")
    # transformer_bce("res18-256")
    # transformer_bce("res18-512")
    # transformer_bce("matlab")
    transformer_bce_gbalance("res18-64")
    transformer_bce_gbalance("res18-128")
    transformer_bce_gbalance("res18-256")
    transformer_bce_gbalance("res18-512")
    # transformer_bce_gbalance("matlab")
    # transformer_fec1("res18-64")
    # transformer_fec1("res18-128")
    # transformer_fec1("res18-256")
    # transformer_fec1("res18-512")
    # transformer_fec5("matlab")
