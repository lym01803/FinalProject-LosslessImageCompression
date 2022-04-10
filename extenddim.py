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

