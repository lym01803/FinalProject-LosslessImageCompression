from regex import E
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import random

import moduleregister
from invertible import InvertibleModule


class NNExtendDim(moduleregister.Register):
    def __init__(self):
        super().__init__()


@NNExtendDim.register
class ExtendDim(InvertibleModule):
    def __init__(self, scale=2):
        super().__init__()
        self.scale = scale

    def forward(self, x, logv):
        shape = x.shape
        scale = self.scale
        x = x.view(shape[0], shape[1], shape[2] // scale, scale, shape[3] // scale, scale)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(shape[0], shape[1] * scale * scale, shape[2] // scale, shape[3] // scale)
        return x, logv

    def backward(self, x):
        shape = x.shape
        scale = self.scale
        x = x.view(shape[0], shape[1] // scale // scale, scale, scale, shape[2], shape[3])
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(shape[0], shape[1] // scale // scale, shape[2] * scale, shape[3] * scale)
        return x 


@moduleregister.Register.register
class Patching(InvertibleModule):
    def __init__(self, H, W, h, w):
        assert (H % h == 0)
        assert (W % w == 0)
        super().__init__()
        self.H = H
        self.W = W
        self.h = h
        self.w = w

    def forward(self, x, logv):
        shape = x.shape
        H, W, h, w = self.H, self.W, self.h, self.w 
        x = x.view(shape[0], shape[1], H // h, h, W // w, w)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(-1, shape[1], h, w)
        return x, logv 

    def backward(self, x):
        shape = x.shape
        H, W, h, w = self.H, self.W, self.h, self.w 
        hh = H // h 
        ww = W // w
        x = x.view(shape[0] // hh // ww, hh, ww, shape[1], shape[2], shape[3])
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(-1, shape[1], H, W)
        return x
    
