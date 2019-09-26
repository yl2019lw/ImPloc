#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torchvision
import torch.nn as nn


class SResnet(nn.Module):

    def __init__(self, depth=50):
        super(SResnet, self).__init__()
        if depth == 50:
            origin = torchvision.models.resnet50(pretrained=True)
            self.proj = nn.Linear(2048, 6)
        elif depth == 34:
            origin = torchvision.models.resnet34(pretrained=True)
            self.proj = nn.Linear(512, 6)
        elif depth == 18:
            origin = torchvision.models.resnet18(pretrained=True)
            self.proj = nn.Linear(512, 6)
        else:
            raise Exception("not supported")

        self.end_layer = -2
        self.features = torch.nn.Sequential(
            *list(origin.children())[:self.end_layer])

    def forward(self, x):
        f = self.features(x)
        f = nn.AdaptiveAvgPool2d(1)(f)
        f = f.view(f.size(0), -1)
        out = self.proj(f)
        return torch.sigmoid(out)


if __name__ == "__main__":
    pass
