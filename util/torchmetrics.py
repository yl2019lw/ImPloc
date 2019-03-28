#!/usr/bin/env python
# -*- coding:  utf-8 -*-

import torch

epsilon = 1e-8
NUM_CLASSES = 6


def compute_f1(precision, recall):
    return 2 * precision * recall / (precision + recall + epsilon)


def example_subset_accuracy(gt, predict):
    ex_equal = (torch.sum(gt == predict, dim=1) == NUM_CLASSES)
    return torch.mean(ex_equal.double())


def example_accuracy(gt, predict):
    ex_and = torch.sum((gt + predict == 2).double(), dim=1)
    ex_or = torch.sum((gt + predict >= 1).double(), dim=1)
    return torch.mean(ex_and / (ex_or + epsilon))


def example_precision(gt, predict):
    ex_and = torch.sum((gt + predict == 2).double(), dim=1)
    ex_predict = torch.sum(predict.double(), dim=1)
    return torch.mean(ex_and / (ex_predict + epsilon))


def example_recall(gt, predict):
    ex_and = torch.sum((gt + predict == 2).double(), dim=1)
    ex_gt = torch.sum(gt.double(), dim=1)
    return torch.mean(ex_and / (ex_gt + epsilon))


def example_f1(gt, predict):
    p = example_precision(gt, predict)
    r = example_recall(gt, predict)
    return (2 * p * r) / (p + r + epsilon)


def _label_quantity(gt, predict):
    tp = torch.sum((gt + predict == 2).double(), dim=0)
    fp = torch.sum((1 - gt + predict == 2).double(), dim=0)
    tn = torch.sum((1 - gt + 1 - predict == 2).double(), dim=0)
    fn = torch.sum((gt + 1 - predict == 2).double(), dim=0)

    return torch.stack((tp, fp, tn, fn), dim=0)


def label_accuracy_macro(gt, predict):
    quantity = _label_quantity(gt, predict)
    tp_tn = torch.add(quantity[0], quantity[2])
    tp_fp_tn_fn = torch.sum(quantity, dim=0)
    return torch.mean(tp_tn / (tp_fp_tn_fn + epsilon))


def label_precision_macro(gt, predict):
    quantity = _label_quantity(gt, predict)
    tp = quantity[0]
    tp_fp = torch.add(quantity[0], quantity[1])
    return torch.mean(tp / (tp_fp + epsilon))


def label_recall_macro(gt, predict):
    quantity = _label_quantity(gt, predict)
    tp = quantity[0]
    tp_fn = torch.add(quantity[0], quantity[3])
    return torch.mean(tp / (tp_fn + epsilon))


def label_f1_macro(gt, predict):
    p = label_precision_macro(gt, predict)
    r = label_recall_macro(gt, predict)
    return (2 * p * r) / (p + r + epsilon)


def label_accuracy_micro(gt, predict):
    quantity = _label_quantity(gt, predict)
    sum_tp, sum_fp, sum_tn, sum_fn = torch.sum(quantity, dim=1)
    return (sum_tp + sum_tn) / (sum_tp + sum_fp + sum_tn + sum_fn + epsilon)


def label_precision_micro(gt, predict):
    quantity = _label_quantity(gt, predict)
    sum_tp, sum_fp, sum_tn, sum_fn = torch.sum(quantity, dim=1)
    return sum_tp / (sum_tp + sum_fp + epsilon)


def label_recall_micro(gt, predict):
    quantity = _label_quantity(gt, predict)
    sum_tp, sum_fp, sum_tn, sum_fn = torch.sum(quantity, dim=1)
    return sum_tp / (sum_tp + sum_fn + epsilon)


def label_f1_micro(gt, predict):
    p = label_precision_micro(gt, predict)
    r = label_recall_micro(gt, predict)
    return (2 * p * r) / (p + r + epsilon)


def single_label_accuracy(gt, predict):
    quantity = _label_quantity(gt, predict)
    tp_tn = torch.add(quantity[0], quantity[2])
    tp_fp_tn_fn = torch.sum(quantity, dim=0)
    return tp_tn / (tp_fp_tn_fn + epsilon)


def single_label_precision(gt, predict):
    quantity = _label_quantity(gt, predict)
    tp = quantity[0]
    tp_fp = torch.add(quantity[0], quantity[1])
    return tp / (tp_fp + epsilon)


def single_label_recall(gt, predict):
    quantity = _label_quantity(gt, predict)
    tp = quantity[0]
    tp_fn = torch.add(quantity[0], quantity[3])
    return tp / (tp_fn + epsilon)


def test():
    a = torch.randint(2, (10, 6))
    b = torch.randint(2, (10, 6))
    p = label_precision_macro(a, b)
    r = label_f1_macro(a, b)
    f = label_f1_macro(a, b)
    print(p, r, f)


if __name__ == "__main__":
    test()
