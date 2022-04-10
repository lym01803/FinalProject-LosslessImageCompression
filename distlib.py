import torch
from torch import nn
from torch.nn import functional as F
from roundlib import NNRound, Round
import moduleregister


class NNDistribution(moduleregister.Register):
    def __init__(self):
        super().__init__()


class Distribution(nn.Module):
    def __init__(self, *args, **kargs):
        super().__init__()

    def sample(self, *args, **kwargs):
        raise NotImplementedError

    def log_prob(self, *args, **kwargs):
        raise NotImplementedError

    def prob(self, *args, **kwargs):
        pass 


@NNDistribution.register
class DLogistic(Distribution):
    def __init__(self, round=None):
        super().__init__()
        if round:
            round_name = round.pop('name')
            self.round = NNRound.get(round_name)(**round)
        else:
            self.round = Round()

    def log_prob(self, x, mean, logscale, nbits=8, eps=1e-8):
        '''
        params:
            x: to calculate Pr(x | mean, logscale), x is assumed to be rounded, which means (x * bins) is integer
            mean: the center of distribution
            logscale: the log-scale. x -> (x - mean) / exp(logscale)
            bins: 1 / (round precision)
        '''
        scale = torch.exp(logscale)
        bins = 2 ** nbits
        x_pos = (x + 0.5 / bins - mean) / scale
        x_neg = (x - 0.5 / bins - mean) / scale
        log_F_pos = F.logsigmoid(x_pos)
        log_F_neg = F.logsigmoid(x_neg)
        logP = log_F_pos + torch.log(1 - torch.exp(log_F_neg - log_F_pos) + eps) # to avoid overflow
        return logP

    def sample(self, mean, logscale, nbits=8):
        '''
        params:
            mean: the center of distribution
            logscale: the log-scale. x -> (x - mean) / exp(logscale)
            bins: 1/ (round precision)
        output:
            a sampled tensor of shape the same as mean / logscale
        '''
        uniform_samples = torch.rand_like(mean)
        std_dlogistic_samples = torch.log(uniform_samples / (1 - uniform_samples))
        affined_samples = std_dlogistic_samples * torch.exp(logscale) + mean
        rounded_samples = self.round(affined_samples, nbits=nbits)
        return rounded_samples

