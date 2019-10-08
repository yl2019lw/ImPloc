#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
import bov
import fisher
import vlad
import joblib
from util import npmetrics


def load_fv(method='bov', fv='fv0'):
    if method == 'bov':
        return bov.load_fv(fv)

    elif method == 'fisher':
        return fisher.load_fv(fv)

    elif method == 'vlad':
        return vlad.load_fv(fv)

    else:
        raise("Not implemented")


def load_kfold_fv(method='bov', fv='fv0', fold=1):
    if method == 'bov':
        return bov.load_kfold_fv(fv, fold)

    elif method == 'fisher':
        return fisher.load_kfold_fv(fv, fold)

    elif method == 'vlad':
        return vlad.load_kfold_fv(fv, fold)

    else:
        raise("Not implemented")


def run_classifier(train_fv, train_label, val_fv, val_label,
                   k, C=1.0, gamma='auto'):
    estimator = SVC(kernel=k, C=C, gamma=gamma)
    classifier = OneVsRestClassifier(estimator, n_jobs=-1)
    classifier.fit(train_fv, train_label)

    val_pd = classifier.predict(val_fv)
    val_f1 = npmetrics.label_f1_macro(val_label, val_pd)
    print("\n---val---", "k:", k, 'C:', C, 'gamma', gamma, 'f1:', val_f1)
    npmetrics.print_metrics(val_label, val_pd)
    return classifier, val_f1


def run_svm(method, fv):
    print("---run svm for method:%s of fv:%s---" % (method, fv))
    train_items, val_items, test_items = load_fv(method, fv)

    train_gene, train_fv, train_label = zip(*train_items)
    val_gene, val_fv, val_label = zip(*val_items)
    test_gene, test_fv, test_label = zip(*test_items)

    scaler = StandardScaler()
    scaler.fit(train_fv)
    train_fv = np.stack(scaler.transform(train_fv))
    val_fv = np.stack(scaler.transform(val_fv))
    test_fv = np.stack(scaler.transform(test_fv))

    train_label = np.stack(train_label)
    val_label = np.stack(val_label)
    test_label = np.stack(test_label)

    best_classifier, best_f1 = run_classifier(
        train_fv, train_label, val_fv, val_label, 'linear')
    joblib.dump(best_classifier,
                "../result/%s_%s_linear.joblib" % (method, fv))

    C_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    gamma_list = ['auto', 0.0001, 0.001, 0.01, 0.1, 1]
    best_k = 'linear'
    best_C = None
    best_gamma = None
    for C in C_list:
        for gamma in gamma_list:
            classifier, val_f1 = run_classifier(
                train_fv, train_label, val_fv, val_label, 'rbf', C, gamma)
            name = '%s_%s_rbf_C%s_g%s' % (method, fv, C, gamma)
            pth = os.path.join("../result/%s.joblib" % name)
            joblib.dump(classifier, pth)

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_classifier = classifier
                best_k = 'rbf'
                best_C = C
                best_gamma = gamma

    test_pd = best_classifier.predict(test_fv)
    print("\n---test res---", "K:", best_k, "C:", best_C, 'gamma', best_gamma)
    npmetrics.print_metrics(test_label, test_pd)


def run_kfold_svm(method, fv, fold=1):
    print("---run svm for method:%s of fv:%s fold:%d---" % (method, fv, fold))
    train_items, val_items, test_items = load_kfold_fv(method, fv, fold)

    train_gene, train_fv, train_label = zip(*train_items)
    val_gene, val_fv, val_label = zip(*val_items)
    test_gene, test_fv, test_label = zip(*test_items)

    scaler = StandardScaler()
    scaler.fit(train_fv)
    train_fv = np.stack(scaler.transform(train_fv))
    val_fv = np.stack(scaler.transform(val_fv))
    test_fv = np.stack(scaler.transform(test_fv))

    train_label = np.stack(train_label)
    val_label = np.stack(val_label)
    test_label = np.stack(test_label)

    best_classifier, best_f1 = run_classifier(
        train_fv, train_label, val_fv, val_label, 'linear')

    tdir = os.path.join("result/%s_%s" % (method, fv))
    if not os.path.exists(tdir):
        os.mkdir(tdir)
    joblib.dump(best_classifier, "%s/fold%d_linear.joblib" % (tdir, fold))

    C_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    gamma_list = ['auto', 0.0001, 0.001, 0.01, 0.1, 1]
    best_k = 'linear'
    best_C = None
    best_gamma = None
    for C in C_list:
        for gamma in gamma_list:
            classifier, val_f1 = run_classifier(
                train_fv, train_label, val_fv, val_label, 'rbf', C, gamma)

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_classifier = classifier
                best_k = 'rbf'
                best_C = C
                best_gamma = gamma

                name = 'fold%d_rbf_C%s_g%s' % (fold, C, gamma)
                pth = os.path.join(tdir, "%s.joblib" % name)
                joblib.dump(classifier, pth)

    test_pd = best_classifier.predict(test_fv)
    path = os.path.join(tdir, "fold%d.txt" % fold)
    from contextlib import redirect_stdout
    with open(path, 'w') as f:
        with redirect_stdout(f):
            print("\n---test res---", "K:", best_k, "C:", best_C,
                  'gamma', best_gamma)
            npmetrics.print_metrics(test_label, test_pd)


def run():
    fvs = ['fv0', 'matlab']
    # methods = ['bov', 'fisher', 'vlad']
    from contextlib import redirect_stdout
    m = 'vlad'
    for fv in fvs:
        path = os.path.join("../result/%s_%s.txt" % (m, fv))
        with open(path, 'w') as f:
            with redirect_stdout(f):
                run_svm(m, fv)
    # run_svm('bov', 'fv0')
    # run_svm('bov', 'matlab')
    # run_svm('fisher', 'fv0')
    # run_svm('fisher', 'matlab')
    # run_svm('vlad', 'fv0')
    # run_svm('vlad', 'matlab')


if __name__ == "__main__":
    folds = list(range(2, 11))
    # fvs = ['fv0', 'matlab']
    fvs = ['matlab']
    # methods = ['bov', 'fisher', 'vlad']
    methods = ['fisher']
    for m in methods:
        for fv in fvs:
            for fold in folds:
                try:
                    run_kfold_svm(m, fv, fold)
                except Exception:
                    continue
    # m = "fisher"
    # fv = "fv0"
    # fold = 1
    # run_kfold_svm(m, fv, fold)
