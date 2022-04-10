import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import random

import moduleregister
from roundlib import NNRound
from nnblock import NNBlock


class NNPrior(moduleregister.Register):
    def __init__(self):
        super().__init__()


@NNPrior.register
class Prior(nn.Module):
    def __init__(self, out_channel, cond_channel, round=None, nn=None):
        '''
        params:
            out_channel: the channel of factorout part
            cond_channel: the channel of conditional part
        Modeling Pr(out|cond)
        '''
        super().__init__()
        self.out_channel = out_channel
        self.cond_channel = cond_channel
        self.round = NNRound.get(round.pop('name'))(**round)
        self.nn_type = NNBlock.get(nn.pop('name'))
        if cond_channel > 0:
            self.NN = self.nn_type(cond_channel, out_channel * 2, **nn)
        else:
            self.NN = self.nn_type(out_channel, out_channel * 2, **nn)

    def forward(self, cond):
        if self.cond_channel > 0:
            params = self.NN(cond)
            mean = params[:, :self.out_channel]
            logscale = params[:, self.out_channel:]
        else:
            params = self.NN(torch.zeros_like(cond))
            mean = params[:, :self.out_channel]
            logscale = params[:, self.out_channel:]
            # mean = torch.zeros_like(cond)
            # logscale = torch.zeros_like(cond)
        return mean, logscale

