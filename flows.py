import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import random
from copy import deepcopy

import moduleregister
from invertible import InvertibleModuleList, Permute
from extenddim import NNExtendDim
from roundlib import NNRound
from distlib import NNDistribution
from couplelib import NNCouple
from priorlib import NNPrior


class NNFlows(moduleregister.Register):
    def __init__(self):
        super().__init__()


@NNFlows.register
class IDFlows(nn.Module):
    def __init__(self, nflows=8, nbits=8, nsplit=3, H=64, W=64, C=3, 
                 couple=None, extenddim=None, prior=None, 
                 distribution=None, round=None):
        '''
        params:
            nflows: number of flows
            nbits: to control round precision
            nsplit: z -> [z1, z2] -> [z1, z2, z3] -> ... -> [z1, z2, ..., z_{nsplit}]
            couple_kwargs:
                channel: the input and output channel number
                split: the split ratio. split to [int(channel * split), channel - int(channel * split)] two parts
                nn_depth: the depth of NN
                growth: the growth param for NN
                nbits: to control the precision of round. the precision is 2 ** (- nbits); can pass nbits either 
                    in __init__ or forward / backward
        '''
        super().__init__()
        self.nflows = nflows
        self.nbits = nbits
        self.nsplit = nsplit
        self.blocks = nn.ModuleList()
        self.latents_shape = []
        self.C = C
        self.H = H
        self.W = W
        self.couple_type = NNCouple.get(couple.pop('name'))
        self.prior_type = NNPrior.get(prior.pop('name'))
        self.extenddim_type = NNExtendDim.get(extenddim.pop('name'))
        self.dist_type = NNDistribution.get(distribution.pop('name'))
        self.round_type = NNRound.get(round.pop('name'))
        channel = C
        h = H
        w = W
        for split_level in range(self.nsplit):
            channel *= 4
            h //= 2
            w //= 2
            flow_module = InvertibleModuleList()
            for flow_idx in range(self.nflows):
                flow_module.append(Permute(dim=channel))
                flow_module.append(self.couple_type(channel=channel, **deepcopy(couple)))
            flow_module.append(Permute(dim=channel))
            if split_level < self.nsplit - 1:
                prior_nn = self.prior_type(channel // 2, channel - channel // 2, **deepcopy(prior))
                self.latents_shape.append((channel // 2, h, w))
                channel -= channel // 2
            else:
                prior_nn = self.prior_type(channel, 0, **deepcopy(prior))
                self.latents_shape.append((channel, h, w))
            self.blocks.append(nn.ModuleDict(dict(
                extend=self.extenddim_type(**deepcopy(extenddim)),
                flows=flow_module,
                prior=prior_nn
            )))
        self.dist = self.dist_type(**distribution)
        self.round = self.round_type(**round)

    def forward(self, x, logv):
        '''
        The output be latents, mean, logscales.
        For compression, you need to encode them further more.
        '''
        latents = []
        means = []
        logscales = []
        for split_level in range(self.nsplit):
            block = self.blocks[split_level]
            x, logv = block['extend'](x, logv)
            x, logv = block['flows'](x, logv)
            if split_level < self.nsplit - 1:
                z, x = x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:]
                mean, logscale = block['prior'](x)
                latents.append(z) # z must be rounded (nbits)
                means.append(mean)
                logscales.append(logscale)
            else:
                z = x
                mean, logscale = block['prior'](x)
                latents.append(z) # z must be rounded (nbits)
                means.append(mean)
                logscales.append(logscale)
        return latents, means, logscales, logv

    def generated_from_noise(self, latents):
        for idx in range(self.nsplit):
            split_level = self.nsplit - 1 - idx
            block = self.blocks[split_level] 
            z = latents[split_level]
            if split_level < self.nsplit - 1:
                mean, logscale = block['prior'](x)
                z = z * torch.exp(logscale) + mean
                z = self.round(z)
                x = torch.cat((z, x), dim=1)
            else: 
                mean, logscale = block['prior'](z)
                z = z * torch.exp(logscale) + mean
                z = self.round(z)
                x = z
            x = block['flows'].backward(x)
            x = block['extend'].backward(x)
        return x

    def log_likelihood(self, latents, means, logscales):
        '''
        The log_likelihood will be calculated datapoint-wise-ly and group-wise-ly;
        '''
        log_Ps = []
        log_prob = torch.zeros((latents[0].shape[0])).to(latents[0].device)
        for idx in range(self.nsplit):
            split_level = idx
            z = latents[split_level]
            mean = means[split_level]
            logscale = logscales[split_level]
            logp = self.dist.log_prob(z, mean, logscale, self.nbits)
            log_Ps.append(torch.mean(logp, dim=(1, 2, 3))) # group-wise and datapoint-wise
            log_prob += torch.sum(logp, dim=(1, 2, 3)) # datapoint-wise; reduce on channel, H, W - dims
        log_prob /= (self.H * self.W * self.C)
        return log_prob, log_Ps

    def inverse(self):
        for block in self.blocks:
            block['extend'].inverse()
            for flow in block['flows']:
                flow.inverse()

    def encode(self, x):
        pass

    def decode(self, x):
        pass

