#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class CNN(nn.Module):
    
    def __init__(self, in_channels, out_channels=50, kernel_size=5):
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.pool = nn.MaxPool1d(kernel_size=in_channels) # We want only the max-over-time right?

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return nn.functional.relu(x)
