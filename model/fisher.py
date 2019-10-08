#!/usr/bin/env python
# -*- coding: utf-8

import numpy as np
import os
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
import fvloader
import matloader
from util import npmetrics


# reduce components for matlab
# def load_gmm(train_data, gmmdir='gmm/fv0', nmix=16):
def load_gmm(train_data, gmmdir='gmm/fv0', nmix=12):
    if not os.path.exists(gmmdir):
        os.mkdir(gmmdir)
    weights_pth = os.path.join(gmmdir, "weights.npy")
    means_pth = os.path.join(gmmdir, "means.npy")
    covs_pth = os.path.join(gmmdir, "covs.npy")
    if os.path.exists(weights_pth):
        print("----load pretrained gmm from %s-----" % gmmdir)
        weights = np.load(weights_pth)
        means = np.load(means_pth)
        covs = np.load(covs_pth)
        return weights, means, covs

    descriptors = []
    for item in train_data:
        gene, nimg, gene_label, timestep = item
        descriptors.append(nimg)

    descriptors = np.concatenate(descriptors)
    # gmm = GaussianMixture(n_components=nmix, covariance_type='diag')
    gmm = GaussianMixture(n_components=nmix)
    print("--------fit gmm-------------", weights_pth)
    gmm.fit(descriptors)

    w = gmm.weights_
    m = gmm.means_
    c = gmm.covariances_

    thr = 0.5 / nmix
    weights = np.float32([v for k, v in zip(range(nmix), w) if w[k] > thr])
    means = np.float32([v for k, v in zip(range(nmix), m) if w[k] > thr])
    covs = np.float32([v for k, v in zip(range(nmix), c) if w[k] > thr])

    np.save(weights_pth, weights)
    np.save(means_pth, means)
    np.save(covs_pth, covs)

    return weights, means, covs


def likelihood_statistics(samples, weights, means, covs):

    def likelihood_moment(x, gamma_tk, moment):
        x_moment = np.power(np.float32(x), moment) if moment > 0 else 1.0
        return x_moment * gamma_tk

    gaussians, s0, s1, s2 = {}, {}, {}, {}
    samples = zip(range(0, len(samples)), samples)

    g = [multivariate_normal(mean=means[k], cov=covs[k], allow_singular=True)
         for k in range(len(weights))]
    for index, x in samples:
        gaussians[index] = np.array([g_k.pdf(x) for g_k in g])

    for k in range(0, len(weights)):
        s0[k], s1[k], s2[k] = 0, 0, 0
        for index, x in samples:
            probabilities = np.multiply(gaussians[index], weights)
            probabilities = probabilities / np.sum(probabilities)
            s0[k] = s0[k] + likelihood_moment(x, probabilities[k], 0)
            s1[k] = s1[k] + likelihood_moment(x, probabilities[k], 1)
            s2[k] = s2[k] + likelihood_moment(x, probabilities[k], 2)

    return s0, s1, s2


def fisher_vector(samples, weights, means, covs):

    def fv_weights(s0, s1, s2, means, covs, w, T):
        return [((s0[k] - T * w[k]) / np.sqrt(w[k]))
                for k in range(0, len(w))]

    def fv_means(s0, s1, s2, means, sigma, w, T):
        return [(s1[k] - means[k] * s0[k]) / (np.sqrt(w[k] * sigma[k]))
                for k in range(0, len(w))]

    def fv_sigma(s0, s1, s2, means, sigma, w, T):
        return [(s2[k] - 2*means[k]*s1[k] + (
                means[k]*means[k]-sigma[k])*s0[k]) / (np.sqrt(2*w[k])*sigma[k])
                for k in range(0, len(w))]

    def normalize(fisher_vector):
        v = np.sqrt(abs(fisher_vector)) * np.sign(fisher_vector)
        return v / np.sqrt(np.dot(v, v))

    s0, s1, s2 = likelihood_statistics(samples, weights, means, covs)

    T = samples.shape[0]
    covs = np.float32([np.diagonal(covs[k]) for k in range(0, covs.shape[0])])
    a = np.array(fv_weights(s0, s1, s2, means, covs, weights, T))
    b = np.array(fv_means(s0, s1, s2, means, covs, weights, T))
    c = np.array(fv_sigma(s0, s1, s2, means, covs, weights, T))
    # print("a", a)
    # print("b", np.concatenate(b).shape)
    # print("c", np.concatenate(c).shape)

    fv = np.concatenate([a, np.concatenate(b), np.concatenate(c)])
    fv = normalize(fv)
    return fv


def data2fisher(dataset, weights, means, covs, fisherdir='fisher/fv0'):
    '''convert origin unpad dataset to fisher vector'''
    if not os.path.exists(fisherdir):
        os.mkdir(fisherdir)

    items = []
    for item in dataset:
        gene, nimg, gene_label, timestep = item
        fv_pth = os.path.join(fisherdir, '%s.npy' % gene)
        if os.path.exists(fv_pth):
            fv = np.load(fv_pth)
        else:
            fv = fisher_vector(nimg, weights, means, covs)
            np.save(fv_pth, fv)
        items.append((gene, fv, gene_label))
    return items


def load_fv(fv='fv0'):
    if fv == 'fv0':
        gmm_data = fvloader.load_train_data(size=0, balance=False)
        weights, means, covs = load_gmm(gmm_data, gmmdir='gmm/fv0')

        train_data = fvloader.load_train_data(size=0, balance=True)
        val_data = fvloader.load_val_data(size=0)
        test_data = fvloader.load_test_data(size=0)

    elif fv == 'matlab':
        gmm_data = matloader.load_train_data(size=0, balance=False)
        weights, means, covs = load_gmm(gmm_data, gmmdir='gmm/matlab')

        train_data = matloader.load_train_data(size=0, balance=True)
        val_data = matloader.load_val_data(size=0)
        test_data = matloader.load_test_data(size=0)

    train_items = data2fisher(train_data, weights, means, covs)
    val_items = data2fisher(val_data, weights, means, covs)
    test_items = data2fisher(test_data, weights, means, covs)

    return train_items, val_items, test_items


def load_kfold_fv(fv='fv0', fold=1):
    if fv == 'fv0':
        gmm_data = fvloader.load_kfold_train_data(fold=fold)
        weights, means, covs = load_gmm(
            gmm_data, gmmdir='gmm/fv0-fold%d' % fold)

        train_data = fvloader.load_kfold_train_data(fold=fold)
        val_data = fvloader.load_kfold_val_data(fold=fold)
        test_data = fvloader.load_kfold_test_data(fold=fold)

        fisherdir = 'fisher/fv0-%d' % fold

    elif fv == 'matlab':
        train_data = matloader.load_kfold_train_data(fold=fold)
        val_data = matloader.load_kfold_val_data(fold=fold)
        gmm_data = train_data
        gmm_data.extend(val_data)

        weights, means, covs = load_gmm(
            gmm_data, gmmdir='gmm/matlab-fold%d' % fold)

        train_data = matloader.load_kfold_train_data(fold=fold)
        val_data = matloader.load_kfold_val_data(fold=fold)
        test_data = matloader.load_kfold_test_data(fold=fold)

        fisherdir = 'fisher/matlab-%d' % fold

    train_items = data2fisher(train_data, weights, means, covs, fisherdir)
    val_items = data2fisher(val_data, weights, means, covs, fisherdir)
    test_items = data2fisher(test_data, weights, means, covs, fisherdir)

    return train_items, val_items, test_items


def svm(fv='fv0'):
    train_items, val_items, test_items = load_fv(fv)
    print("train items", len(train_items))
    print("val items", len(val_items))
    print("test items", len(test_items))

    train_gene, train_fv, train_label = zip(*train_items)
    val_gene, val_fv, val_label = zip(*val_items)
    test_gene, test_fv, test_label = zip(*test_items)

    print("-------run svm for fisher--------", fv)
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
            print("\n---svm for fisher---", "k:", k, 'b:', b, 'f1:', val_f1)
            npmetrics.print_metrics(val_label, val_pd)
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_classifier = classifier
                best_k = k
                best_b = b

    test_pd = best_classifier.predict(test_fv)
    print("\n---svm for fisher test result---", "k:", best_k, "b:", best_b)
    npmetrics.print_metrics(test_label, test_pd)


def test_fisher(size=0):
    weights, means, covs = load_gmm()
    print("weights", weights.shape)
    print("means", means.shape)
    print("covriances", covs.shape)
    train_data = fvloader.load_train_data(size=size, balance=False)
    for item in train_data:
        gene, nimg, gene_label, timestep = item
        fv = fisher_vector(nimg, weights, means, covs)
        print("gene", gene, "count", timestep, "fv", fv.shape)


if __name__ == "__main__":
    # test_fisher()
    # svm()
    svm(fv='matlab')
