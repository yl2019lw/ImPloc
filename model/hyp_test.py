#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import spm1d
import scipy
import os
import time
from scipy import linalg
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
import torch
import sys
sys.path.append("../")
from util import torch_util

NUM_CLASSES = 6


def load_si_result():
    normal = pd.read_csv('normal_si.txt')
    cancer = pd.read_csv('cancer_si.txt')

    n_gene = normal.gene.unique()
    c_gene = cancer.gene.unique()

    gene_list = np.intersect1d(n_gene, c_gene)

    normal_data = {}
    cancer_data = {}

    for gene in gene_list:
        normal_data[gene] = normal[normal.gene == gene]
        cancer_data[gene] = cancer[cancer.gene == gene]

    return gene_list, normal_data, cancer_data


def filter_mislocation(gene_list, normal, cancer):
    '''filter mislocation by mean of instance score'''

    def fmt(s):
        tmp = [x.strip(' \'[]') for x in s.split()]
        r = [float(x) for x in tmp if len(x)]
        return r

    mis_genes = []

    for g in gene_list:
        nprob = np.stack(normal[g].prob.apply(fmt))
        cprob = np.stack(cancer[g].prob.apply(fmt))
        nu = np.mean(nprob, axis=0)
        cu = np.mean(cprob, axis=0)
        npd = (nu >= 0.5).astype(np.int)
        cpd = (cu >= 0.5).astype(np.int)
        if np.array_equal(npd, cpd):
            pass
        else:
            mis_genes.append(g)
    return mis_genes


def bag_filter_mislocation(gene_list):
    '''filter mislocation by bag predict'''
    torch.nn.Module.dump_patches = True
    model_name = "ubploss/transformer_res18-128_size0_autoloss_alpha16"
    model_dir = os.path.join("./modeldir/%s" % model_name)
    model_pth = os.path.join(model_dir, "model.pth")

    model = torch.load(model_pth).cuda()

    mis_genes = []
    for g in gene_list:
        print("filter gene %s" % g)
        npd = predict_normal_gene(model, g)
        cpd = predict_cancer_gene(model, g)
        const = [0] * 6
        if np.array_equal(const, npd):
            continue
        if np.array_equal(const, cpd):
            continue
        if np.array_equal(npd, cpd):
            continue
        print("found mislocation gene %s" % g)
        mis_genes.append(g)
    return mis_genes


def predict_normal_gene(model, gene):
    nfv_dir = '/ndata/longwei/hpa/tissuefv/res18_128'
    ngp = os.path.join(nfv_dir, '%s.npy' % gene)
    if not os.path.exists(ngp):
        print("normal fv for %s not exists" % gene)
        return [0] * 6

    model.eval()
    with torch.no_grad():

        npy = np.expand_dims(np.load(ngp), axis=0)
        nin = torch.from_numpy(npy).type(torch.cuda.FloatTensor)
        npd = model(nin)
        test_npd = torch_util.threshold_tensor_batch(npd)

        time.sleep(1)

        return test_npd.data.cpu().numpy()


def predict_cancer_gene(model, gene):
    cfv_dir = '/ndata/longwei/hpa/cancerfv_4tissue'
    cgp = os.path.join(cfv_dir, '%s.npy' % gene)
    if not os.path.exists(cgp):
        print("cancer fv for %s not exists" % gene)
        return [0] * 6

    model.eval()
    with torch.no_grad():
        cpy = np.expand_dims(np.load(cgp), axis=0)
        cin = torch.from_numpy(cpy).type(torch.cuda.FloatTensor)
        cpd = model(cin)
        test_cpd = torch_util.threshold_tensor_batch(cpd)

        return test_cpd.data.cpu().numpy()


def hotelling_test_gene(normal, cancer):

    def fmt(s):
        tmp = [x.strip(' \'[]') for x in s.split()]
        r = [float(x) for x in tmp if len(x)]
        return r

    nprob = np.stack(normal.prob.apply(fmt))
    cprob = np.stack(cancer.prob.apply(fmt))
    print("nprob", nprob.shape, "cprob", cprob.shape)
    try:
        t2 = spm1d.stats.hotellings2(nprob, cprob)
    except Exception as e:
        print('except', nprob.shape, cprob.shape)
        return 1.0
    p = t2.inference().p
    return p


def hotelling_test_all():
    gene_list, normal, cancer = load_si_result()

    gene_list = filter_mislocation(gene_list, normal, cancer)

    ps = []
    for g in gene_list:
        print("---test for %s---" % g)
        p = hotelling_test_gene(normal[g], cancer[g])
        ps.append(p)

    df = pd.DataFrame({'gene': gene_list, 'p': ps}).sort_values('p')
    df.to_csv("hotelling.csv", header=True, index=False, sep=',')


def dr(inputs):
    pca = decomposition.PCA(n_components=6)
    x_std = StandardScaler().fit_transform(inputs)
    return pca.fit_transform(x_std)[:, 0]


def pca(data, dims=1):
    data -= data.mean(axis=0)
    R = np.cov(data, rowvar=False)
    evals, evecs = linalg.eigh(R)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    evals = evals[idx]
    evecs = evecs[:, :dims]
    return np.dot(evecs.T, data.T).T


def t_test_gene(gene, normal, cancer):

    def fmt(s):
        tmp = [x.strip(' \'[]') for x in s.split()]
        r = [float(x) for x in tmp if len(x)]
        return r

    try:
        nprob = np.stack(normal.prob.apply(fmt))
        cprob = np.stack(cancer.prob.apply(fmt))

        nprob = pca(nprob)
        cprob = pca(cprob)
        np.savetxt('pca/%s_normal.txt' % gene, nprob)
        np.savetxt('pca/%s_cancer.txt' % gene, cprob)
        # print('nprob', nprob)
        # print('cprob', cprob)
        _, p = scipy.stats.ttest_ind(nprob, cprob)
        # t2 = spm1d.stats.ttest2(nprob, cprob, equal_var=False)
        # p = t2.inference(two_tailed=True).p
        return p
    except Exception as e:
        print(e)
        return 1.0


def t_test_all():
    gene_list, normal, cancer = load_si_result()

    ps = []
    for g in gene_list:
        print("---test for %s---" % g)
        p = t_test_gene(g, normal[g], cancer[g])
        ps.append(p)

    df = pd.DataFrame({'gene': gene_list, 'p': ps}).sort_values('p')
    df.to_csv("t_test.csv", header=True, index=False, sep=',')


def t_test_separate_gene(gene, normal, cancer):

    def fmt(s):
        tmp = [x.strip(' \'[]') for x in s.split()]
        r = [float(x) for x in tmp if len(x)]
        return r

    nprob = np.stack(normal.prob.apply(fmt))
    cprob = np.stack(cancer.prob.apply(fmt))

    ps = []
    for i in range(NUM_CLASSES):
        _, p = scipy.stats.ttest_ind(nprob[:, i], cprob[:, i])
        ps.append(p)
    return ps


def t_test_separate_all():
    gene_list, normal, cancer = load_si_result()

    # gene_list = filter_mislocation(gene_list, normal, cancer)
    gene_list = bag_filter_mislocation(gene_list)

    ps_list = []
    sig_genes = []
    sig_ps = []
    for g in gene_list:
        print("---test for %s---" % g)
        ps = t_test_separate_gene(g, normal[g], cancer[g])
        ps_list.append(ps)
        if np.all(np.array(ps) < 0.05):
            sig_genes.append(g)
            sig_ps.append(np.mean(ps))

    mp = [np.mean(x) for x in ps_list]
    df = pd.DataFrame({'gene': gene_list, 'p': ps_list,
                       'mp': mp}).sort_values('mp')
    df.to_csv("bagfilter_t_test_separate0.05.csv",
              header=True, index=False, sep=',')

    sig_df = pd.DataFrame({'gene': sig_genes, 'p': sig_ps}).sort_values('p')
    sig_df.to_csv("bagfilter_t_test_separate_sig0.05.csv",
                  header=True, index=False, sep=',')


def mislocation():
    genes = ['ENSG00000128342', 'ENSG00000023734', 'ENSG00000105705',
             'ENSG00000111011', 'ENSG00000116863', 'ENSG00000197265',
             'ENSG00000160208', 'ENSG00000159352', 'ENSG00000115539',
             'ENSG00000163946', 'ENSG00000116161']

    gene_list, normal_data, cancer_data = load_si_result()

    def fmt(s):
        tmp = [x.strip(' \'[]') for x in s.split()]
        r = [float(x) for x in tmp if len(x)]
        return r

    for g in genes:
        normal = normal_data[g]
        cancer = cancer_data[g]
        nprob = np.stack(normal.prob.apply(fmt))
        cprob = np.stack(cancer.prob.apply(fmt))
        nu = np.mean(nprob, axis=0)
        cu = np.mean(cprob, axis=0)

        npd = (nu > 0.5).astype(np.int)
        cpd = (cu > 0.5).astype(np.int)

        print("gene:", g)
        print("normal:", npd)
        print("cancer:", cpd)


def bag_mislocation():
    genes = ['ENSG00000128342', 'ENSG00000023734', 'ENSG00000105705',
             'ENSG00000111011', 'ENSG00000116863', 'ENSG00000197265',
             'ENSG00000160208', 'ENSG00000159352', 'ENSG00000115539',
             'ENSG00000163946', 'ENSG00000116161']

    torch.nn.Module.dump_patches = True
    model_name = "ubploss/transformer_res18-128_size0_autoloss_alpha16"
    model_dir = os.path.join("./modeldir/%s" % model_name)
    model_pth = os.path.join(model_dir, "model.pth")

    model = torch.load(model_pth).cuda()

    cfv_dir = '/ndata/longwei/hpa/cancerfv_4tissue'
    nfv_dir = '/ndata/longwei/hpa/tissuefv/res18_128'

    model.eval()
    with torch.no_grad():
        for g in genes:
            ngp = os.path.join(nfv_dir, '%s.npy' % g)
            cgp = os.path.join(cfv_dir, '%s.npy' % g)
            if not os.path.exists(ngp):
                print("normal fv for %s not exists" % g)
                continue
            if not os.path.exists(cgp):
                print("cancer fv for %s not exists" % g)
                continue

            npy = np.expand_dims(np.load(ngp), axis=0)
            nin = torch.from_numpy(npy).type(torch.cuda.FloatTensor)
            npd = model(nin)
            test_npd = torch_util.threshold_tensor_batch(npd)

            time.sleep(1)
            cpy = np.expand_dims(np.load(cgp), axis=0)
            cin = torch.from_numpy(cpy).type(torch.cuda.FloatTensor)
            cpd = model(cin)
            test_cpd = torch_util.threshold_tensor_batch(cpd)

            print("gene", g)
            print("normal pd", test_npd.data.cpu().numpy())
            print("cancer pd", test_cpd.data.cpu().numpy())


if __name__ == "__main__":
    # hotelling_test_all()
    # t_test_all()
    t_test_separate_all()
    # mislocation()
    # bag_mislocation()
