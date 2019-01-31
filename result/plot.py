#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_aggres18():
    df = pd.read_csv('aggres18.csv').set_index('agg')

    fig, ax = plt.subplots()

    methods = np.array(df.index)
    metrics = np.array(df.columns)
    xticks = np.array(range(len(methods)))
    xlabels = methods

    for m in metrics:
        y = np.array(df[m])
        ax.plot(xticks, y, label=m)

    ax.set_title("agg for res18 fv")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.tick_params(axis='x', which='major', pad=10)
    ax.set_ylabel('metrics')
    # tlabels = ax.xaxis.get_ticklabels()
    # [t.set_rotation(30) for t in tlabels]

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout(rect=[0, 0, 0.8, 1.0])
    plt.show()


def plot_agg():
    rdf = pd.read_csv('aggres18.csv').set_index('agg')
    sdf = pd.read_csv('aggslf.csv').set_index('agg')
    dfs = [rdf, sdf]

    fig, axes = plt.subplots(1, 2, figsize=(20, 5))
    for i in range(2):
        ax = axes[i]
        df = dfs[i]

        methods = np.array(df.index)
        metrics = np.array(df.columns)
        xticks = np.array(range(len(methods)))
        xlabels = methods

        colors = ['#3399CC', '#33CC99', '#6600FF',
                  '#000000', '#CC3366', '#666666',
                  '#993300', '#FF0099', '#FFCC33']
        markers = ['o', '<', '>', '*', '+', 'x', '^', 's', 'v']
        linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-']

        for j, m in enumerate(metrics):
            y = np.array(df[m])
            ax.plot(xticks, y, label=m, linewidth=2,
                    color=colors[j], marker=markers[j],
                    linestyle=linestyles[j])

        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
        ax.set_xlim(left=-0.2, right=4.2)
        ax.set_ylim(bottom=-0.05, top=1.05)
        ax.set_yticks(np.linspace(0, 1, 11))
        ax.tick_params(axis='x', width=2, pad=10, labelcolor='b', top='off')
        # tlabels = ax.xaxis.get_ticklabels()
        # [t.set_rotation(10) for t in tlabels]
        # ax.set_ylabel('metrics')
        if i == 0:
            ax.set_title("Aggregation for Res18-128")
            ax.tick_params(axis='y', right='off')
        else:
            ax.set_title("Aggregation for SLFs")
            ax.tick_params(axis='y', left='off')
            ax.yaxis.set_ticks_position('right')
        ax.title.set_position((0.5, 1.03))

    legend_labels = [
        'subset_acc', 'ex_accuracy', 'ex_precision', 'ex_recall', 'ex_f1',
        'label_accuracy', 'label_precision', 'label_recall', 'label_f1']
    plt.legend(labels=legend_labels, title='Metrics',
               loc='upper left', bbox_to_anchor=(1.05, 1.02),
               shadow=True, fancybox=True, numpoints=1)
    plt.tight_layout(rect=[0, 0, 0.86, 1])
    # plt.show()
    plt.savefig('agg.eps')


def plot_loss():
    df = pd.read_csv('loss.csv').set_index('loss')
    methods = np.array(df.index)
    metrics = np.array(df.columns)

    fig, ax = plt.subplots(figsize=(20, 5))
    bar_width = 0.13
    opacity = 0.8
    # colors = ['b', 'g', 'r', 'c', 'm', 'k']
    colors = ['#3399CC', '#33CC99', '#660066', '#996633', '#CC3366', '#CCFF66']

    pos = np.arange(len(metrics))
    for i, m in enumerate(methods):
        plt.bar(pos + i*bar_width, df.loc[m], bar_width,
                alpha=opacity,  color=colors[i],
                label=m)

    legend_labels = [
        r'$BCE$', r'$Focal(\gamma=0.5)$',
        r'$Focal(\gamma=1)$', r'$Focal(\gamma=2)$',
        r'$Focal(\gamma=3)$', r'$Penalty$']
    metrics_labels = [
        'subset_accuracy', 'example_accuracy', 'example_precision',
        'example_recall', 'example_f1',
        'label_accuracy', 'label_precision', 'label_recall', 'label_f1']
    plt.xticks(pos+0.4, metrics_labels)
    ax.tick_params(axis='x', length=0, pad=8)

    ax.set_xlim(left=-0.2, right=9.0)
    ax.set_ylim(bottom=0.0, top=1.15)
    leg = plt.legend(labels=legend_labels, loc='best', ncol=6,
                     mode='expand', shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.4)
    plt.tight_layout()
    # plt.legend(labels=legend_labels, title=r'Different Losses',
    #            loc='upper left', bbox_to_anchor=(1, 1))
    # plt.tight_layout(rect=[0, 0, 0.85, 1.0])
    # plt.show()
    plt.savefig('loss.eps')


def plot_selfv():
    df = pd.read_csv('res18fv.csv').set_index('fv')
    fvs = np.array(df.index)
    metrics = np.array(df.columns)

    fig, ax = plt.subplots(figsize=(20, 5))
    bar_width = 0.16
    opacity = 0.8
    # colors = ['b', 'g', 'r', 'c', 'k']
    colors = ['#3399CC', '#33CC99', '#660066', '#996633', '#CC3366']

    pos = np.arange(len(metrics))
    for i, m in enumerate(fvs):
        plt.bar(pos + i*bar_width, df.loc[m], bar_width,
                alpha=opacity, color=colors[i],
                label=m)

    metrics_labels = [
        'subset_accuracy', 'example_accuracy', 'example_precision',
        'example_recall', 'example_f1',
        'label_accuracy', 'label_precision', 'label_recall', 'label_f1']
    plt.xticks(pos+0.4, metrics_labels)
    ax.tick_params(axis='x', length=0, pad=8)

    ax.set_xlim(left=-0.2, right=9.0)
    ax.set_ylim(bottom=0.0, top=1.15)
    leg = plt.legend(loc='best', ncol=5, mode='expand',
                     shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.4)
    plt.tight_layout()
    # plt.legend(title=r'Different features',
    #            loc='upper left', bbox_to_anchor=(1, 1))
    # plt.tight_layout(rect=[0, 0, 0.85, 1.0])
    # plt.show()
    plt.savefig('selfv.eps')


if __name__ == "__main__":
    # plot_aggres18()
    plot_agg()
    # plot_loss()
    # plot_selfv()
