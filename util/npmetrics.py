#!/usr/bin/env python
# -*- coding: utf-8 -*-

# multi label metrics
import numpy as np

epsilon = 1e-8


def compute_f1(precision, recall):
    return 2 * precision * recall / (precision + recall + epsilon)


def example_subset_accuracy(gt, predict):
    ex_equal = np.all(np.equal(gt, predict), axis=1).astype("float32")
    return np.mean(ex_equal)


def example_accuracy(gt, predict):
    ex_and = np.sum(np.logical_and(gt, predict), axis=1).astype("float32")
    ex_or = np.sum(np.logical_or(gt, predict), axis=1).astype("float32")
    return np.mean((ex_and + epsilon) / (ex_or + epsilon))


def example_precision(gt, predict):
    ex_and = np.sum(np.logical_and(gt, predict), axis=1).astype("float32")
    ex_predict = np.sum(predict, axis=1).astype("float32")
    return np.mean((ex_and + epsilon) / (ex_predict + epsilon))


def example_recall(gt, predict):
    ex_and = np.sum(np.logical_and(gt, predict), axis=1).astype("float32")
    ex_gt = np.sum(gt, axis=1).astype("float32")
    return np.mean((ex_and + epsilon) / (ex_gt + epsilon))


def example_f1(gt, predict):
    p = example_precision(gt, predict)
    r = example_recall(gt, predict)
    return (2 * p * r) / (p + r + epsilon)


def _label_quantity(gt, predict):
    tp = np.sum(np.logical_and(gt, predict), axis=0)
    fp = np.sum(np.logical_and(1-gt, predict), axis=0)
    tn = np.sum(np.logical_and(1-gt, 1-predict), axis=0)
    fn = np.sum(np.logical_and(gt, 1-predict), axis=0)
    return np.stack([tp, fp, tn, fn], axis=0).astype("float")


def label_accuracy_macro(gt, predict):
    quantity = _label_quantity(gt, predict)
    tp_tn = np.add(quantity[0], quantity[2])
    tp_fp_tn_fn = np.sum(quantity, axis=0)
    return np.mean((tp_tn + epsilon) / (tp_fp_tn_fn + epsilon))


def label_precision_macro(gt, predict):
    quantity = _label_quantity(gt, predict)
    tp = quantity[0]
    tp_fp = np.add(quantity[0], quantity[1])
    return np.mean((tp + epsilon) / (tp_fp + epsilon))


def label_recall_macro(gt, predict):
    quantity = _label_quantity(gt, predict)
    tp = quantity[0]
    tp_fn = np.add(quantity[0], quantity[3])
    return np.mean((tp + epsilon) / (tp_fn + epsilon))


def label_f1_macro(gt, predict):
    p = label_precision_macro(gt, predict)
    r = label_recall_macro(gt, predict)
    return (2 * p * r) / (p + r + epsilon)


def label_accuracy_micro(gt, predict):
    quantity = _label_quantity(gt, predict)
    sum_tp, sum_fp, sum_tn, sum_fn = np.sum(quantity, axis=1)
    return (sum_tp + sum_tn + epsilon) / (
            sum_tp + sum_fp + sum_tn + sum_fn + epsilon)


def label_precision_micro(gt, predict):
    quantity = _label_quantity(gt, predict)
    sum_tp, sum_fp, sum_tn, sum_fn = np.sum(quantity, axis=1)
    return (sum_fp + epsilon) / (sum_tp + sum_fp + epsilon)


def label_recall_micro(gt, predict):
    quantity = _label_quantity(gt, predict)
    sum_tp, sum_fp, sum_tn, sum_fn = np.sum(quantity, axis=1)
    return (sum_tp + epsilon) / (sum_tp + sum_fn + epsilon)


def label_f1_micro(gt, predict):
    p = label_precision_micro(gt, predict)
    r = label_recall_micro(gt, predict)
    return (2 * p * r) / (p + r + epsilon)


def single_label_accuracy(gt, predict):
    quantity = _label_quantity(gt, predict)
    tp_tn = np.add(quantity[0], quantity[2])
    tp_fp_tn_fn = np.sum(quantity, axis=0)
    return (tp_tn + epsilon) / (tp_fp_tn_fn + epsilon)


def single_label_precision(gt, predict):
    quantity = _label_quantity(gt, predict)
    tp = quantity[0]
    tp_fp = np.add(quantity[0], quantity[1])
    return (tp + epsilon) / (tp_fp + epsilon)


def single_label_recall(gt, predict):
    quantity = _label_quantity(gt, predict)
    tp = quantity[0]
    tp_fn = np.add(quantity[0], quantity[3])
    return (tp + epsilon) / (tp_fn + epsilon)


def print_metrics(gt, predict):
    ex_subset_acc = example_subset_accuracy(gt, predict)
    ex_acc = example_accuracy(gt, predict)
    ex_precision = example_precision(gt, predict)
    ex_recall = example_recall(gt, predict)
    ex_f1 = compute_f1(ex_precision, ex_recall)

    lab_acc_macro = label_accuracy_macro(gt, predict)
    lab_precision_macro = label_precision_macro(gt, predict)
    lab_recall_macro = label_recall_macro(gt, predict)
    lab_f1_macro = compute_f1(lab_precision_macro, lab_recall_macro)

    lab_acc_micro = label_accuracy_micro(gt, predict)
    lab_precision_micro = label_precision_micro(gt, predict)
    lab_recall_micro = label_recall_micro(gt, predict)
    lab_f1_micro = compute_f1(lab_precision_micro, lab_recall_micro)

    print("example_subset_accuracy:", ex_subset_acc)
    print("example_accuracy:", ex_acc)
    print("example_precision:", ex_precision)
    print("example_recall:", ex_recall)
    print("example_f1:", ex_f1)

    print("label_accuracy_macro:", lab_acc_macro)
    print("label_precision_macro:", lab_precision_macro)
    print("label_recall_macro:", lab_recall_macro)
    print("label_f1_macro:", lab_f1_macro)

    print("label_accuracy_micro:", lab_acc_micro)
    print("label_precision_micro:", lab_precision_micro)
    print("label_recall_micro:", lab_recall_micro)
    print("label_f1_micro:", lab_f1_micro)


def write_metrics(gt, predict, path):
    ex_subset_acc = example_subset_accuracy(gt, predict)
    ex_acc = example_accuracy(gt, predict)
    ex_precision = example_precision(gt, predict)
    ex_recall = example_recall(gt, predict)
    ex_f1 = compute_f1(ex_precision, ex_recall)

    lab_acc_macro = label_accuracy_macro(gt, predict)
    lab_precision_macro = label_precision_macro(gt, predict)
    lab_recall_macro = label_recall_macro(gt, predict)
    lab_f1_macro = compute_f1(lab_precision_macro, lab_recall_macro)

    lab_acc_micro = label_accuracy_micro(gt, predict)
    lab_precision_micro = label_precision_micro(gt, predict)
    lab_recall_micro = label_recall_micro(gt, predict)
    lab_f1_micro = compute_f1(lab_precision_micro, lab_recall_micro)

    with open(path, 'w') as f:
        f.write("example_subset_accuracy:   %.4f\n" % ex_subset_acc)
        f.write("example_accuracy:          %.4f\n" % ex_acc)
        f.write("example_precision:         %.4f\n" % ex_precision)
        f.write("example_recall:            %.4f\n" % ex_recall)
        f.write("example_f1:                %.4f\n" % ex_f1)

        f.write("label_accuracy_macro:      %.4f\n" % lab_acc_macro)
        f.write("label_precision_macro:     %.4f\n" % lab_precision_macro)
        f.write("label_recall_macro:        %.4f\n" % lab_recall_macro)
        f.write("label_f1_macro:            %.4f\n" % lab_f1_macro)

        f.write("label_accuracy_micro:      %.4f\n" % lab_acc_micro)
        f.write("label_precision_micro:     %.4f\n" % lab_precision_micro)
        f.write("label_recall_micro:        %.4f\n" % lab_recall_micro)
        f.write("label_f1_micro:            %.4f\n" % lab_f1_micro)


if __name__ == "__main__":
    pass
