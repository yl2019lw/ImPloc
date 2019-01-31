#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch


class FineTuneModel(torch.nn.Module):

    def __init__(self, original_model):
        super(FineTuneModel, self).__init__()

        self.features = torch.nn.Sequential(
                        *list(original_model.children())[:-2])

        for p in self.features.parameters():
            p.require_grad = False

    def forward(self, x):
        f = self.features(x)
        f = torch.nn.AdaptiveAvgPool2d(1)(f)
        return f.view(f.size(0), -1)
