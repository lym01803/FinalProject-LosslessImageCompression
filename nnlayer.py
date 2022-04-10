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

import moduleregister
from activate import ActivateFunc


class NNLayer(moduleregister.Register):
    def __init__(self):
        super().__init__()


@NNLayer.register
class DenseLayer(nn.Module):
    def __init__(self, i_channel, o_channel, act='ReLU'):
        '''
        params:
            i_channel: number of input channels
            growth_channel: output_channels - input_channels, means the additional channels output by this layer
        '''
        super().__init__()
        self.i_channel = i_channel
        self.o_channel = o_channel
        if act == 'ReLU':
            self.act = nn.ReLU()
        elif act == 'Tanh':
            self.act = nn.Tanh()
        elif act == 'LeakyReLU':
            self.act = nn.LeakyReLU()
        else:
            self.act = ActivateFunc.get(act)()
        
        self.layers = nn.Sequential(
            nn.Conv2d(i_channel, i_channel, kernel_size=1),
            nn.Conv2d(i_channel, o_channel - i_channel, kernel_size=3, padding=1),
            self.act
        )

    def forward(self, x):
        h = self.layers(x)
        h = torch.cat((x, h), dim=1)
        return h

