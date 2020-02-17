#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class CNN(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)

    def forward(self, x):
        return nn.functional.relu(self.pool(self.conv(x)))
