#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from util import npmetrics
from util import torchmetrics


NUM_CLASSES = 6
epsilon = 1e-8


def torch_metrics(gt, predict, writer, step, mode="val"):
    ex_subset_acc = npmetrics.example_subset_accuracy(gt, predict)
    ex_acc = npmetrics.example_accuracy(gt, predict)
    ex_precision = npmetrics.example_precision(gt, predict)
    ex_recall = npmetrics.example_recall(gt, predict)
    ex_f1 = npmetrics.compute_f1(ex_precision, ex_recall)

    lab_acc_macro = npmetrics.label_accuracy_macro(gt, predict)
    lab_precision_macro = npmetrics.label_precision_macro(gt, predict)
    lab_recall_macro = npmetrics.label_recall_macro(gt, predict)
    lab_f1_macro = npmetrics.compute_f1(lab_precision_macro, lab_recall_macro)

    lab_acc_micro = npmetrics.label_accuracy_micro(gt, predict)
    lab_precision_micro = npmetrics.label_precision_micro(gt, predict)
    lab_recall_micro = npmetrics.label_recall_micro(gt, predict)
    lab_f1_micro = npmetrics.compute_f1(lab_precision_micro, lab_recall_micro)

    writer.add_scalar("%s subset acc" % mode, ex_subset_acc, step)
    writer.add_scalar("%s example acc" % mode, ex_acc, step)
    writer.add_scalar("%s example precision" % mode, ex_precision, step)
    writer.add_scalar("%s example recall" % mode, ex_recall, step)
    writer.add_scalar("%s example f1" % mode, ex_f1, step)

    writer.add_scalar("%s label acc macro" % mode,
                      lab_acc_macro, step)
    writer.add_scalar("%s label precision macro" % mode,
                      lab_precision_macro, step)
    writer.add_scalar("%s label recall macro" % mode,
                      lab_recall_macro, step)
    writer.add_scalar("%s label f1 macro" % mode, lab_f1_macro, step)

    writer.add_scalar("%s label acc micro" % mode,
                      lab_acc_micro, step)
    writer.add_scalar("%s label precision micro" % mode,
                      lab_precision_micro, step)
    writer.add_scalar("%s label recall micro" % mode,
                      lab_recall_micro, step)
    writer.add_scalar("%s label f1 micro" % mode,
                      lab_f1_micro, step)

    sl_acc = npmetrics.single_label_accuracy(gt, predict)
    sl_precision = npmetrics.single_label_precision(gt, predict)
    sl_recall = npmetrics.single_label_recall(gt, predict)
    for i in range(NUM_CLASSES):
        writer.add_scalar("%s sl_%d_acc" % (mode, i),
                          sl_acc[i], step)
        writer.add_scalar("%s sl_%d_precision" % (mode, i),
                          sl_precision[i], step)
        writer.add_scalar("%s sl_%d_recall" % (mode, i),
                          sl_recall[i], step)
    return lab_f1_macro


def threshold_tensor_batch(pd, base=0.5):
    '''make sure at least one label for batch'''
    p_max = torch.max(pd, dim=1)[0]
    pivot = torch.cuda.FloatTensor([base]).expand_as(p_max)
    threshold = torch.min(p_max, pivot)
    pd_threshold = torch.ge(pd, threshold.unsqueeze(dim=1))
    return pd_threshold


def threshold_pd(pd, base=0.5):
    '''make sure at least one label for one example'''
    p_max = torch.max(pd)
    pivot = torch.cuda.FloatTensor([base])
    threshold = torch.min(p_max, pivot)
    pd_threshold = torch.ge(pd, threshold)
    return pd_threshold


class FECLoss(nn.Module):
    '''auto weighted loss, called Penalty loss in paper'''
    def __init__(self, alpha=100, gamma=1,
                 reduction='mean', thr=0.5):
        '''
        alpha controls weights between bce & penalty.
        gamma controls penalty level.
        p > thr as positive, otherwise negative.
        '''
        super(FECLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.thr = thr

    def agree_mask(self, p, y):
        '''return 0-1 agree mask, p > 0.5 for y = 1, p < 0.5 for y = 0'''
        sign = (p - self.thr) * (y - self.thr)
        return torch.sigmoid(1e8 * sign)

    def mask_tp(self, p, y):
        '''tp is y == 1 and p agree with y'''
        return y * self.agree

    def mask_fp(self, p, y):
        '''fp is y == 0 and p not agree with y'''
        return (1 - y) * (1 - self.agree)

    def mask_tn(self, p, y):
        '''tn is y == 0 and p agree with y'''
        return (1 - y) * self.agree

    def mask_fn(self, p, y):
        '''fn is y == 1 and p not agree with y'''
        return y * (1 - self.agree)

    def forward(self, p, y):
        oloss = F.binary_cross_entropy(p, y, reduction='none')

        self.agree = self.agree_mask(p, y)
        nsample, nlabel = y.shape

        tp_ind = self.mask_tp(p, y)
        fp_ind = self.mask_fp(p, y)
        tn_ind = self.mask_tn(p, y)
        fn_ind = self.mask_fn(p, y)

        tp = torch.sum(tp_ind, dim=0)
        fp = torch.sum(fp_ind, dim=0)
        tn = torch.sum(tn_ind, dim=0)
        fn = torch.sum(fn_ind, dim=0)

        fp_coef = fp / (tn + 1)
        fn_coef = fn / (tp + 1)
        fp_w = fp_coef * fp_ind
        fn_w = fn_coef * fn_ind

        penalty = fp_w ** self.gamma + fn_w ** self.gamma
        weights = 1 + self.alpha * penalty / nsample

        w_loss = weights * oloss

        if self.reduction == 'none':
            return w_loss
        else:
            return torch.mean(w_loss)


class FbetaLoss(nn.Module):
    '''add fbeta loss bias to recall'''

    def __init__(self, beta=0.8, bce=False, factor=1.0):
        super(FbetaLoss, self).__init__()
        self.beta = beta
        self.bce = bce
        self.factor = factor

    def forward(self, p, y):
        floss = self.compute_fbetaloss(p, y).float()
        if self.bce:
            oloss = F.binary_cross_entropy(p, y, reduction='elementwise_mean')
            floss = oloss + floss * self.factor
        return floss

    def compute_fbetaloss(self, p, y):
        bs = self.beta * self.beta
        ma_r = torchmetrics.label_recall_macro(y, p)
        ma_p = torchmetrics.label_precision_macro(y, p)
        f_beta = ((1 + bs) * ma_r * ma_p) / (bs * ma_p + ma_r + epsilon)
        return 1.0 - f_beta


class RecallLoss(nn.Module):
    '''add recall to loss'''

    def __init__(self, bce=False, factor=1.0):
        super(RecallLoss, self).__init__()
        self.bce = bce
        self.factor = factor

    def forward(self, p, y):
        rloss = self.compute_rloss(p, y).float()
        if self.bce:
            oloss = F.binary_cross_entropy(p, y, reduction='elementwise_mean')
            rloss = oloss + rloss * self.factor
        return rloss

    def compute_rloss(self, p, y):
        ma_r = torchmetrics.label_recall_macro(y, p)
        return 1.0 - ma_r


class PrecisionLoss(nn.Module):
    '''add precision to loss'''

    def __init__(self, bce=False, factor=1.0):
        super(PrecisionLoss, self).__init__()
        self.bce = bce
        self.factor = factor

    def forward(self, p, y):
        ploss = self.compute_ploss(p, y).float()
        if self.bce:
            oloss = F.binary_cross_entropy(p, y, reduction='elementwise_mean')
            ploss = oloss + ploss * self.factor
        return ploss

    def compute_ploss(self, p, y):
        ma_p = torchmetrics.label_precision_macro(y, p)
        return 1.0 - ma_p


class F1Loss(nn.Module):
    '''add f1 term to loss'''

    def __init__(self, bce=False, factor=1.0):
        super(F1Loss, self).__init__()
        self.bce = bce
        self.factor = factor

    def forward(self, p, y):
        floss = self.compute_floss(p, y).float()
        if self.bce:
            oloss = F.binary_cross_entropy(p, y, reduction='elementwise_mean')
            floss = oloss + floss * self.factor
        return floss

    def compute_floss(self, p, y):
        ex_f1 = torchmetrics.example_f1(y, p)
        ma_f1 = torchmetrics.label_f1_macro(y, p)
        mi_f1 = torchmetrics.label_f1_micro(y, p)
        return 3 - ex_f1 - ma_f1 - mi_f1


class SignLoss(nn.Module):
    '''add sign term to loss(y & p consistency)'''

    def __init__(self, threshold=0.5, bce=False, factor=1.0):
        super(SignLoss, self).__init__()
        self.thr = threshold
        self.bce = bce
        self.factor = factor

    def forward(self, p, y):
        bound = self.thr * self.thr
        sloss = torch.mean(bound - (p - self.thr) * (y - self.thr))
        if self.bce:
            oloss = F.binary_cross_entropy(p, y, reduction='elementwise_mean')
            sloss = oloss + sloss * self.factor
        return sloss


class SFocalLoss(nn.Module):
    '''Simple Focal Loss without alpha t'''
    def __init__(self, gamma=2, reduction="elementwise_mean"):
        super(SFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, p, y):
        pt = y * p + (1 - y) * (1 - p)
        pt = pt.clamp(epsilon, 1.0 - epsilon)

        w = (1 - pt).pow(self.gamma)
        w = w.clamp(epsilon, 10000.0)
        loss = -w * pt.log() / 2
        if self.reduction == 'none':
            return loss
        else:
            return torch.mean(loss)
        # return torch.mean(torch.sum(loss, dim=1))


class FocalLoss(nn.Module):

    def __init__(self, freq, gamma=2):
        super(FocalLoss, self).__init__()
        self.freq = freq
        self.gamma = gamma

    def forward(self, p, y):
        pt = y * p + (1 - y) * (1 - p)
        pt = pt.clamp(epsilon, 1.0 - epsilon)

        w = (1 - pt).pow(self.gamma)

        tfreq = torch.from_numpy(self.freq).type(torch.cuda.FloatTensor)
        # at = y * ((1.0/tfreq) ** self.gamma) + (
        #         1 - y) * ((1.0/(1 - tfreq)) ** self.gamma)
        at = y * (1.0/tfreq) + (1 - y) * (1.0/(1 - tfreq))

        w = at * w
        w = w.clamp(epsilon, 10000.0)
        # print("label", y)
        # print("at", at)
        # print("weight", w)

        loss = -w * pt.log() / 2
        return torch.mean(torch.sum(loss, dim=1))
