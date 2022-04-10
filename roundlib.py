import torch
from torch import nn

import moduleregister


class NNRound(moduleregister.Register):
    def __init__(self):
        super().__init__()


class BaseRound(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        y = torch.round(x)
        return x + (y - x).detach()


@NNRound.register
class Round(nn.Module):
    def __init__(self, nbits=None):
        super().__init__()
        self.nbits = nbits
        self.round = BaseRound()
    
    def forward(self, x, nbits=None):
        n_bit = nbits or self.nbits or 8
        bins = 2 ** n_bit
        x = self.round(x * bins) / bins
        return x
