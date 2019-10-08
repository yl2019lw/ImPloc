#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import fvloader
import matloader
from util import npmetrics
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
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


def vlad(nimg, kmeans):
    pd = kmeans.predict(nimg)
    centers = kmeans.cluster_centers_
    k = kmeans.n_clusters

    m, d = nimg.shape
    ret = np.zeros([k, d])
    for i in range(k):
        if np.sum(pd == i) > 0:
            ret[i] = np.sum(nimg[pd == i, :] - centers[i], axis=0)
    ret = ret.flatten()
    ret = np.sign(ret) * np.sqrt(np.abs(ret))
    ret = ret / np.sqrt(np.dot(ret, ret))
    return ret


def data2vlad(dataset, kmeans, vladdir='vlad/fv0'):
    if not os.path.exists(vladdir):
        os.mkdir(vladdir)
    items = []
    for item in dataset:
        gene, nimg, gene_label, timestep = item
        fv_pth = os.path.join(vladdir, '%s.npy' % gene)
        if os.path.exists(fv_pth):
            fv = np.load(fv_pth)
        else:
            fv = vlad(nimg, kmeans)
            np.save(fv_pth, fv)
        items.append((gene, fv, gene_label))
    return items


def load_fv(fv='fv0'):
    if fv == 'fv0':
        vlad_data = fvloader.load_train_data(size=0, balance=False)
        kmeans = cluster(vlad_data)

        train_data = fvloader.load_train_data(size=0, balance=True)
        val_data = fvloader.load_val_data(size=0)
        test_data = fvloader.load_test_data(size=0)
        vladdir = 'vlad/fv0'

    elif fv == 'matlab':
        vlad_data = matloader.load_train_data(size=0, balance=False)
        kmeans = cluster(vlad_data)

        train_data = matloader.load_train_data(size=0, balance=True)
        val_data = matloader.load_val_data(size=0)
        test_data = matloader.load_test_data(size=0)
        vladdir = 'vlad/matlab'

    train_items = data2vlad(train_data, kmeans, vladdir=vladdir)
    val_items = data2vlad(val_data, kmeans, vladdir=vladdir)
    test_items = data2vlad(test_data, kmeans, vladdir=vladdir)

    return train_items, val_items, test_items


def load_kfold_fv(fv='fv0', fold=1):
    if fv == 'fv0':
        vlad_data = fvloader.load_kfold_train_data(fold=fold)
        kmeans = cluster(vlad_data)

        train_data = fvloader.load_kfold_train_data(fold=fold)
        val_data = fvloader.load_kfold_val_data(fold=fold)
        test_data = fvloader.load_kfold_test_data(fold=fold)
        vladdir = 'vlad/fv0-fold%d' % fold

    elif fv == 'matlab':
        vlad_data = matloader.load_kfold_train_data(fold=fold)
        kmeans = cluster(vlad_data)

        train_data = matloader.load_kfold_train_data(fold=fold)
        val_data = matloader.load_kfold_val_data(fold=fold)
        test_data = matloader.load_kfold_test_data(fold=fold)
        vladdir = 'vlad/matlab-fold%d' % fold

    train_items = data2vlad(train_data, kmeans, vladdir=vladdir)
    val_items = data2vlad(val_data, kmeans, vladdir=vladdir)
    test_items = data2vlad(test_data, kmeans, vladdir=vladdir)

    return train_items, val_items, test_items


def svm(fv='fv0'):
    train_items, val_items, test_items = load_fv(fv)
    print("train items", len(train_items))
    print("val items", len(val_items))
    print("test items", len(test_items))

    train_gene, train_fv, train_label = zip(*train_items)
    val_gene, val_fv, val_label = zip(*val_items)
    test_gene, test_fv, test_label = zip(*test_items)

    print("-------run svm for vlad--------", fv)
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
            print("\n---svm for vlad---", "k:", k, 'b:', b, 'f1:', val_f1)
            npmetrics.print_metrics(val_label, val_pd)
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_classifier = classifier
                best_k = k
                best_b = b

    test_pd = best_classifier.predict(test_fv)
    print("\n---svm for vlad test result---", "k:", best_k, "b:", best_b)
    npmetrics.print_metrics(test_label, test_pd)


def test_vlad(size=0):
    kmeans = cluster()
    train_data = fvloader.load_train_data(size=size, balance=False)
    for item in train_data:
        gene, nimg, gene_label, timestep = item
        fv = vlad(nimg, kmeans)
        print("gene", gene, "count", timestep, "fv", fv.shape)


if __name__ == "__main__":
    # svm()
    svm(fv='matlab')
