import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import normal_, _no_grad_fill_
from torch.nn.parameter import Parameter

import numpy as np
import math
import os
import sys
import random
from copy import deepcopy

import moduleregister
from nnlayer import NNLayer


class NNBlock(moduleregister.Register):
    def __init__(self):
        super().__init__()


@NNBlock.register
class DenseBlock(nn.Module):
    def __init__(self, i_channel, o_channel, layer, growth_channel=512, depth=8):
        '''
        params:
            i_channel: number of input channels
            o_channel: number of output channels
            growth_channel: means: i_channel -> ... -> (i_channel + growth_channel) -> o_channel
            depth: the number of denselayer(s)
            act: str, can be 'ReLU' or 'Tanh' or 'LeakyReLU'
        '''
        super().__init__()
        self.i_channel = i_channel
        self.o_channel = o_channel
        self.growth_channel = growth_channel
        self.depth = depth
        self.layer_type = NNLayer.get(layer.pop('name'))
        self.layers = nn.ModuleList()
        channel = i_channel
        for idx in range(depth):
            growth = (idx + 1) * growth_channel // depth - idx * growth_channel // depth
            self.layers.append(self.layer_type(i_channel=channel, o_channel=channel+growth, **deepcopy(layer)))
            channel += growth
        assert (channel == (i_channel + growth_channel))
        self.layers.append(nn.Conv2d(i_channel + growth_channel, o_channel, 1))

        _no_grad_fill_(self.layers[-1].weight.data, 0)
        _no_grad_fill_(self.layers[-1].bias.data, 0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x 
