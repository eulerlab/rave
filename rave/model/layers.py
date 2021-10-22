import torch
import torch.nn as nn
import numpy as np


class SkipConnection(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module(x) + x


class ResidualLayer(torch.nn.Module):
    def __init__(self, num_feature, dropout):
        super().__init__()
        block = nn.Sequential(
            nn.BatchNorm1d(num_feature),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(num_feature, num_feature, bias=False),
            nn.BatchNorm1d(num_feature),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(num_feature, num_feature, bias=False),
        )

        # initialize near identity:
        block[-1].weight.data *= 1e-3

        self.layer = SkipConnection(block)

    def forward(self, x):
        return self.layer(x)


class ResidualLayerV2(torch.nn.Module):
    def __init__(self, num_feature_in, num_feature_out, dropout):
        super().__init__()
        block = nn.Sequential(
            nn.BatchNorm1d(num_feature_in),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(num_feature_in, num_feature_out, bias=False),
            nn.BatchNorm1d(num_feature_out),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(num_feature_out, num_feature_out, bias=False),
        )

        # initialize near identity:
        block[-1].weight.data *= 1e-3

        self.layer = block

    def forward(self, x):
        return self.layer(x)
