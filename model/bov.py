#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import fvloader
import matloader
from util import npmetrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

NCLUSTER = 64


def _all_descriptors(train_data):
    descriptors = []
    for item in train_data:
        gene, nimg, gene_label, timestep = item
        descriptors.append(nimg)

    descriptors = np.concatenate(descriptors)
    return descriptors


def cluster(train_data):
    descriptors = _all_descriptors(train_data)
    kmeans = KMeans(n_clusters=NCLUSTER)
    return kmeans.fit(descriptors)


def bow(nimg, kmeans):
    ret = np.zeros(NCLUSTER)
    index = kmeans.predict(nimg)
    for i in index:
        ret[i] += 1
    return ret


def data2bov(dataset, kmeans, bovdir='bov/fv0'):
    if not os.path.exists(bovdir):
        os.mkdir(bovdir)
    items = []
    for item in dataset:
        gene, nimg, gene_label, timestep = item
        fv_pth = os.path.join(bovdir, '%s.npy' % gene)
        if os.path.exists(fv_pth):
            fv = np.load(fv_pth)
        else:
            fv = bow(nimg, kmeans)
            np.save(fv_pth, fv)
        items.append((gene, fv, gene_label))
    return items


def load_fv(fv='fv0'):
    if fv == 'fv0':
        kmean_data = fvloader.load_train_data(size=0, balance=False)
        kmeans = cluster(kmean_data)

        train_data = fvloader.load_train_data(size=0, balance=True)
        val_data = fvloader.load_val_data(size=0)
        test_data = fvloader.load_test_data(size=0)
        bovdir = 'bov/fv0'

    elif fv == 'matlab':
        kmean_data = matloader.load_train_data(size=0, balance=False)
        kmeans = cluster(kmean_data)

        train_data = matloader.load_train_data(size=0, balance=True)
        val_data = matloader.load_val_data(size=0)
        test_data = matloader.load_test_data(size=0)
        bovdir = 'bov/matlab'

    train_items = data2bov(train_data, kmeans, bovdir=bovdir)
    val_items = data2bov(val_data, kmeans, bovdir=bovdir)
    test_items = data2bov(test_data, kmeans, bovdir=bovdir)

    return train_items, val_items, test_items


def load_kfold_fv(fv='fv0', fold=1):
    if fv == 'fv0':
        kmean_data = fvloader.load_kfold_train_data(fold=fold)
        kmeans = cluster(kmean_data)

        train_data = fvloader.load_kfold_train_data(fold=fold)
        val_data = fvloader.load_kfold_val_data(fold=fold)
        test_data = fvloader.load_kfold_test_data(fold=fold)
        bovdir = 'bov/fv0-fold%d' % fold

    elif fv == 'matlab':
        kmean_data = matloader.load_kfold_train_data(fold=fold)
        kmeans = cluster(kmean_data)

        train_data = matloader.load_kfold_train_data(fold=fold)
        val_data = matloader.load_kfold_val_data(fold=fold)
        test_data = matloader.load_kfold_test_data(fold=fold)
        bovdir = 'bov/matlab-%d' % fold

    train_items = data2bov(train_data, kmeans, bovdir=bovdir)
    val_items = data2bov(val_data, kmeans, bovdir=bovdir)
    test_items = data2bov(test_data, kmeans, bovdir=bovdir)

    return train_items, val_items, test_items


def svm(fv='fv0'):
    train_items, val_items, test_items = load_fv(fv)
    print("train items", len(train_items))
    print("val items", len(val_items))
    print("test items", len(test_items))

    train_gene, train_fv, train_label = zip(*train_items)
    val_gene, val_fv, val_label = zip(*val_items)
    test_gene, test_fv, test_label = zip(*test_items)

    print("-------run svm for bov--------", fv)
    scaler = StandardScaler()
    scaler.fit(train_fv)
    train_fv = np.stack(scaler.transform(train_fv))
    val_fv = np.stack(scaler.transform(val_fv))
    test_fv = np.stack(scaler.transform(test_fv))

    train_label = np.stack(train_label)
    val_label = np.stack(val_label)
    test_label = np.stack(test_label)

    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    class_weights = ['balanced', None]
    best_f1 = 0.0
    best_classifier = None
    best_k = None
    best_b = None
    for k in kernels:
        for b in class_weights:
            estimator = SVC(kernel=k, class_weight=b)
            classifier = OneVsRestClassifier(estimator, n_jobs=-1)
            classifier.fit(train_fv, train_label)

            val_pd = classifier.predict(val_fv)
            val_f1 = npmetrics.label_f1_macro(val_label, val_pd)
            print("\n---svm for bov---", "k:", k, 'b:', b, 'f1:', val_f1)
            npmetrics.print_metrics(val_label, val_pd)
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_classifier = classifier
                best_k = k
                best_b = b

    test_pd = best_classifier.predict(test_fv)
    print("\n---svm for bov test result---", "k:", best_k, "b:", best_b)
    npmetrics.print_metrics(test_label, test_pd)


def test_bow(size=0):
    kmeans = cluster()
    train_data = fvloader.load_train_data(size=size, balance=False)
    for item in train_data:
        gene, nimg, gene_label, timestep = item
        fv = bow(nimg, kmeans)
        print("gene", gene, "count", timestep, "fv", fv.shape)


if __name__ == "__main__":
    # svm(fv='fv0')
    svm(fv='matlab')
    # test_bow()
