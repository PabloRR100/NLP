#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import torch.nn as nn


class Highway(nn.Module):
    '''
    Args:
        embed_size: word embedding dimensionality 
    '''
    def __init__(self, embed_size=50, bias=True):
        super(Highway, self).__init__()
        self.W_proj = nn.Linear(embed_size, embed_size, bias=bias)
        self.W_gate = nn.Linear(embed_size, embed_size, bias=bias)

    def forward(self, x_conv):
        # print('x_conv_shape: ', x_conv.shape)
        x_proj = torch.relu(self.W_proj(x_conv))
        # print('x_proj: ', x_proj)
        # print('x_proj: ', x_proj.shape)
        x_gate = torch.sigmoid(self.W_gate(x_conv))
        # print('x_gate: ', x_gate)
        # print('x_gate: ', x_gate.shape)
        x_highway = x_gate * x_proj + (1-x_gate)*x_conv
        # print('x_highway: ', x_highway.shape)
        return x_highway

if __name__ == '__main__':
    batch_size = 2
    embed_size = 3
    W_proj = np.array([[0.5 for _ in range(embed_size)] for _ in range(embed_size)])
    W_gate = np.array([[0.1 for _ in range(embed_size)] for _ in range(embed_size)])
    input_x = np.array([[0.8 for _ in range(embed_size)] for _ in range(batch_size)])

    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    def relu(x):
        return np.maximum(x, 0)

    x_proj = relu(np.dot(input_x, W_proj))
    x_gate = sigmoid(np.dot(input_x, W_gate))
    x_high = x_gate * x_proj + (1-x_gate) * input_x
    # print('x_proj: ', x_proj)
    # print('x_gate: ', x_gate)
    # print('x_highway: ', x_high)

    net = Highway(3, bias=False)
    net.W_proj.weight = nn.Parameter(torch.Tensor(W_proj))
    net.W_gate.weight = nn.Parameter(torch.Tensor(W_gate))
    x = torch.Tensor(input_x)
    net(x)

