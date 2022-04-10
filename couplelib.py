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

from roundlib import NNRound
from invertible import InvertibleModule
import moduleregister
from nnblock import NNBlock

class NNCouple(moduleregister.Register):
    def __init__(self):
        super().__init__()
    

@NNCouple.register
class AdditiveCouple(InvertibleModule):
    def __init__(self, channel, split=0.75, nn=None, round=None):
        '''
        params:
            channel: the input and output channel number
            split: the split ratio. split to [int(channel * split), channel - int(channel * split)] two parts
            nn_depth: the depth of NN
            growth: the growth param for NN
            nbits: to control the precision of round. the precision is 2 ** (- nbits); can pass nbits either 
                in __init__ or forward / backward
        '''
        super().__init__()
        self.channel = channel
        self.split = split
        a_ch, b_ch = int(channel * split), channel - int(channel * split)
        self.a_ch = a_ch
        self.b_ch = b_ch
        # z_a = x_a, z_b = x_b + t(x_a)
        self.nn_type = NNBlock.get(nn.pop('name'))
        self.dense = self.nn_type(i_channel=a_ch, o_channel=b_ch, **nn)
        self.round_type = NNRound.get(round.pop('name'))
        self.round = self.round_type(**round)

    def forward(self, x, logv, nbits=None):
        xa, xb = x[:, :self.a_ch], x[:, self.a_ch:]
        txa = self.round(self.dense(xa))
        za = xa 
        zb = xb + txa 
        z = torch.cat([za, zb], dim=1)
        return z, logv
        
    def backward(self, z, nbits=None):
        za, zb = z[:, :self.a_ch], z[:, self.a_ch:]
        xa = za
        txa = self.round(self.dense(xa))
        xb = zb - txa 
        x = torch.cat([xa, xb], dim=1)
        return x
