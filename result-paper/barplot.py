#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def plot_loss():
    df = pd.read_csv('loss.csv').set_index('loss')
    legend_labels = [
        r'$BCE$', r'$Focal(\gamma=0.5)$',
        r'$Focal(\gamma=1)$', r'$Focal(\gamma=2)$',
        r'$Focal(\gamma=3)$', r'$Penalty$']
    methods = np.array(df.index)
    metrics = np.array(df.columns)

    fig, ax = plt.subplots(figsize=(20, 5))
    plt.style.use('ggplot')
    bar_width = 0.13
    opacity = 0.8

    # hatchs = ['-', '+', 'x', '\\', '*', 'o', '.', '\/']

    pos = np.arange(len(metrics))
    for i, m in enumerate(methods):
        plt.bar(pos + i*bar_width, df.loc[m], bar_width,
                alpha=opacity, label=m)

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
    plt.savefig("loss.eps")


def plot_agg_res18():
    df = pd.read_csv('aggres18.csv').set_index('agg')
    methods = np.array(df.index)
    metrics = np.array(df.columns)

    legend_labels = methods

    fig, ax = plt.subplots(figsize=(20, 5))
    plt.style.use('ggplot')
    bar_width = 0.17
    opacity = 0.8

    # hatchs = ['-', '+', 'x', '\\', '*', 'o', '.', '\/']

    pos = np.arange(len(metrics))
    for i, m in enumerate(methods):
        plt.bar(pos + i*bar_width, df.loc[m], bar_width,
                alpha=opacity, label=m)

    metrics_labels = [
        'subset_accuracy', 'example_accuracy', 'example_precision',
        'example_recall', 'example_f1',
        'label_accuracy', 'label_precision', 'label_recall', 'label_f1']

    plt.xticks(pos+0.4, metrics_labels)
    ax.tick_params(axis='x', length=0, pad=9)

    ax.set_xlim(left=-0.2, right=9.0)
    ax.set_ylim(bottom=0.0, top=1.05)

    leg = plt.legend(labels=legend_labels, loc='best', ncol=len(legend_labels),
                     mode='expand', shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.4)
    plt.tight_layout()
    plt.savefig("agg_res18.eps")


def plot_agg_slf():
    df = pd.read_csv('aggslf.csv').set_index('agg')
    methods = np.array(df.index)
    metrics = np.array(df.columns)

    legend_labels = methods

    fig, ax = plt.subplots(figsize=(20, 5))
    plt.style.use('ggplot')
    bar_width = 0.17
    opacity = 0.8

    # hatchs = ['-', '+', 'x', '\\', '*', 'o', '.', '\/']

    pos = np.arange(len(metrics))
    for i, m in enumerate(methods):
        plt.bar(pos + i*bar_width, df.loc[m], bar_width,
                alpha=opacity, label=m)

    metrics_labels = [
        'subset_accuracy', 'example_accuracy', 'example_precision',
        'example_recall', 'example_f1',
        'label_accuracy', 'label_precision', 'label_recall', 'label_f1']

    plt.xticks(pos+0.4, metrics_labels)
    ax.tick_params(axis='x', length=0, pad=9)

    ax.set_xlim(left=-0.2, right=9.0)
    ax.set_ylim(bottom=0.0, top=1.05)

    leg = plt.legend(labels=legend_labels, loc='best', ncol=len(legend_labels),
                     mode='expand', shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.4)
    plt.tight_layout()
    plt.savefig("agg_slf.eps")


def plot_selfv():
    df = pd.read_csv('res18fv.csv').set_index('fv')
    fvs = np.array(df.index)
    metrics = np.array(df.columns)

    fig, ax = plt.subplots(figsize=(20, 5))
    bar_width = 0.16
    opacity = 0.8
    plt.style.use('ggplot')

    pos = np.arange(len(metrics))
    for i, m in enumerate(fvs):
        plt.bar(pos + i*bar_width, df.loc[m], bar_width,
                alpha=opacity, label=m)

    metrics_labels = [
        'subset_accuracy', 'example_accuracy', 'example_precision',
        'example_recall', 'example_f1',
        'label_accuracy', 'label_precision', 'label_recall', 'label_f1']
    plt.xticks(pos+0.4, metrics_labels)
    ax.tick_params(axis='x', length=0, pad=8)

    ax.set_xlim(left=-0.2, right=9.0)
    ax.set_ylim(bottom=0.2, top=0.95)
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
    # plot_loss()
    # plot_agg_res18()
    # plot_agg_slf()
    plot_selfv()
