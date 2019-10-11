#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def plot_agg(fv="slf"):
    if fv == 'slf':
        basedir = "agg_slf"
    else:
        basedir = "agg_resnet"
    df = pd.read_csv('%s/result.csv' % basedir).set_index('sid')

    methods = df['model'].unique()
    metrics = np.array(df.columns.drop('fold'))

    meandf = df.groupby('model').mean()
    mindf = meandf - df.groupby('model').min()
    maxdf = df.groupby('model').max() - meandf

    legend_labels = methods

    fig, ax = plt.subplots(figsize=(18, 5))
    plt.style.use('ggplot')
    bar_width = 0.18
    opacity = 0.8

    # hatchs = ['-', '+', 'x', '\\', '*', 'o', '.', '\/']

    pos = np.arange(len(metrics))
    for i, m in enumerate(methods):
        yerr = np.stack([mindf.loc[m][metrics], maxdf.loc[m][metrics]])
        plt.bar(pos + i*bar_width, meandf.loc[m][metrics], bar_width,
                alpha=opacity, label=m, yerr=yerr)

    metrics_labels = metrics
    # metrics_labels = [
    #     'subset_accuracy', 'example_accuracy', 'example_precision',
    #     'example_recall', 'example_f1',
    #     'label_accuracy', 'label_precision', 'label_recall', 'label_f1']

    plt.xticks(pos+0.4, metrics_labels)
    ax.tick_params(axis='x', length=0, pad=9)

    ax.set_xlim(left=-0.17, right=8.91)
    ax.set_ylim(bottom=0.2, top=1.05)

    leg = plt.legend(labels=legend_labels, loc='best', ncol=len(legend_labels),
                     mode='expand', shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.4)
    plt.tight_layout()
    plt.savefig("%s/%s-mean.eps" % (basedir, fv))


if __name__ == "__main__":
    plot_agg("slf")
    plot_agg("resnet")
