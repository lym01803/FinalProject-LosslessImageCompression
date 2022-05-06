import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import random
from copy import deepcopy

import math

import moduleregister
from invertible import InvertibleModuleList, Permute
from extenddim import ExtendDim, NNExtendDim, Patching
from roundlib import NNRound, Round
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
        extendscale = extenddim.get('scale')
        for split_level in range(self.nsplit):
            channel *= extendscale * extendscale
            h //= extendscale
            w //= extendscale
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

    def generated_from_latents(self, latents):
        for idx in range(self.nsplit):
            split_level = self.nsplit - 1 - idx
            block = self.blocks[split_level] 
            z = latents[split_level]
            if split_level < self.nsplit - 1:
                x = torch.cat((z, x), dim=1)
            else: 
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


@NNFlows.register
class TwoLevelFlows(nn.Module):
    def __init__(self, H, W, C, pad, fine_flows, rough_flows, batchsize=256, nbits=8):
        """
        For convenience, assume that fine and rough has nsplit=1.
        The batchsize is used for the fined patches.
        """
        super().__init__()
        self.H = H + pad[0]
        self.W = W + pad[1]
        self.C = C
        self.pad = pad
        self.pad2d = nn.ReplicationPad2d(padding=(0, pad[1], 0, pad[0]))
        self.fine = NNFlows.get(fine_flows.pop('name'))(**fine_flows)
        self.rough = NNFlows.get(rough_flows.pop('name'))(**rough_flows)
        self.pool = nn.AdaptiveAvgPool2d((self.rough.H, self.rough.W))
        self.invpool = nn.AdaptiveAvgPool2d((self.H, self.W))
        self.patching = Patching(self.H, self.W, self.fine.H, self.fine.W)
        self.latents_shape = [deepcopy(self.rough.latents_shape[0]), deepcopy(self.fine.latents_shape[0])]
        self.latents_shape[1] = (self.latents_shape[1][0] * (self.H // self.fine.H) * (self.W // self.fine.W), 
                                 self.latents_shape[1][1], self.latents_shape[1][2]) # channel - dim
        self.batchsize = batchsize
        self.ratio = (self.H // self.fine.H, self.W // self.fine.W)
        self.round = Round(nbits=nbits)

    def forward(self, x, logv, train=True):
        '''
        To save cuda memory, need to do backward inside this function.
        '''
        x = self.pad2d(x)
        rx = self.round(self.pool(x))
        fx = x - self.invpool(rx)
        rlatent, rmean, rlogscale, logv = self.rough.forward(rx, logv)
        flatents, fmeans, flogscales = [], [], []
        logP, logPs = self.rough.log_likelihood(rlatent, rmean, rlogscale)
        loss = torch.mean(-logP, dim=0)
        if train:
            loss.backward()
        bpd1 = loss.item() / math.log(2)
        rlatent, rmean, rlogscale = rlatent[0], rmean[0], rlogscale[0]
        # flatents, fmeans, flogscales = [], [], []
        px, logv = self.patching(fx, logv)
        nums = px.shape[0]
        bs = self.batchsize
        bpd2 = 0.
        for i in range((nums + bs - 1) // bs):
            bpx = px[bs * i : bs * i + bs]
            flatent, fmean, flogscale, logv = self.fine.forward(bpx, logv)
            logP, logPs = self.fine.log_likelihood(flatent, fmean, flogscale)
            loss = torch.mean(-logP, dim=0)
            if train:
                loss.backward()
            bpd2 += loss.item() / math.log(2) * bpx.shape[0]
            flatents.append(flatent[0])
            fmeans.append(fmean[0])
            flogscales.append(flogscale[0])
        bpd2 /= px.shape[0]
        bpd = (bpd1 * self.rough.H * self.rough.W + bpd2 * self.H * self.W) / (self.H - self.pad[0]) / (self.W - self.pad[1])
        latents = [rlatent, torch.cat(flatent, dim=0)]
        means = [rmean, torch.cat(fmeans, dim=0)]
        logscales = [rlogscale, torch.cat(flogscales, dim=0)]
        return latents, means, logscales, bpd, bpd1, bpd2, logv

    def generated_from_noise(self, latents):
        '''
        latents = [rlatent, flatent]
        rlatent will be in shape (bs, c, rh, rw)
        flatent will be in shape (bs, c*num, fh, fw)
        '''
        bs = latents[0].shape[0]
        # print(latents[0].shape, latents[1].shape)
        rx = self.rough.generated_from_noise([latents[0]])
        fx = []
        flatents = latents[1].view(-1, self.C * 4, latents[1].shape[2], latents[1].shape[3])
        # print(flatents.shape)
        # input()
        bs = self.batchsize
        nums = flatents.shape[0]
        for i in range((nums + bs - 1) // bs):
            fx.append(self.fine.generated_from_noise(
                [flatents[i * bs : i * bs + bs]]
            ))
        fx = torch.cat(fx, dim=0)
        fx = self.patching.backward(fx)
        x = self.invpool(rx) + fx 
        x = x[:, :, :x.shape[2] - self.pad[0], :x.shape[3] - self.pad[1]]
        return x

    def inverse(self):
        self.fine.inverse()
        self.rough.inverse()


@NNFlows.register
class ConditionalFlows(IDFlows):
    def __init__(self, conv_for_cond=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # x = \bar(x) + res
        # \bar(x) will be the condition fed into nn_prior
        # nn_prior will have additional input channels
        ch = self.C
        self.conv_for_cond = conv_for_cond
        if conv_for_cond:
            self.convs = nn.ModuleList()
        for split_level in range(self.nsplit):
            block = self.blocks[split_level]
            extend = block['extend']
            extend: ExtendDim
            scale = extend.scale    
            ch *= scale * scale
            prior = block['prior']
            block['prior'] = self.prior_type(prior.out_channel,
                                             prior.cond_channel + ch if prior.cond_channel > 0 else prior.out_channel + ch, 
                                             **deepcopy(kwargs.get('prior')))
            if conv_for_cond:
                self.convs.append(
                    nn.Conv2d(ch // scale // scale, ch, 4, 2, 1)
                )
        
    def forward(self, x, logv, cond):
        latents = []
        means = []
        logscales = []
        for split_level in range(self.nsplit):
            block = self.blocks[split_level]
            x, logv = block['extend'](x, logv)
            if not self.conv_for_cond:
                cond, _ = block['extend'](cond, None)
            else:
                cond = self.convs[split_level](cond)
            x, logv = block['flows'](x, logv)
            if split_level < self.nsplit - 1:
                z, x = x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:]
                mean, logscale = block['prior'](torch.cat((x, cond), dim=1))
                latents.append(z) # z must be rounded (nbits)
                means.append(mean)
                logscales.append(logscale)
            else:
                z = x
                mean, logscale = block['prior'](torch.cat((torch.zeros_like(x), cond), dim=1))
                latents.append(z) # z must be rounded (nbits)
                means.append(mean)
                logscales.append(logscale)
        return latents, means, logscales, logv

    def generated_from_noise(self, latents, cond):
        if not self.conv_for_cond:
            for idx in range(self.nsplit):
                cond, _ = self.blocks[idx]['extend'](cond, None)
        else:
            conds = [self.convs[0](cond)]
            for idx in range(1, self.nsplit):
                conds.append(self.convs[idx](conds[-1]))
        for idx in range(self.nsplit):
            split_level = self.nsplit - 1 - idx
            block = self.blocks[split_level] 
            z = latents[split_level]
            if split_level < self.nsplit - 1:
                if not self.conv_for_cond:
                    mean, logscale = block['prior'](torch.cat((x, cond), dim=1))
                else:
                    mean, logscale = block['prior'](torch.cat((x, conds[split_level]), dim=1))
                z = z * torch.exp(logscale) + mean
                z = self.round(z)
                x = torch.cat((z, x), dim=1)
            else: 
                if not self.conv_for_cond:
                    mean, logscale = block['prior'](torch.cat((torch.zeros_like(z), cond), dim=1))
                else:
                    mean, logscale = block['prior'](torch.cat((torch.zeros_like(z), conds[split_level]), dim=1))
                z = z * torch.exp(logscale) + mean
                z = self.round(z)
                x = z
            x = block['flows'].backward(x)
            x = block['extend'].backward(x)
            if not self.conv_for_cond:
                cond = block['extend'].backward(cond)
        return x

