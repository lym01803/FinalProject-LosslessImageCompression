from colorama import reinit
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

import moduleregister
from torch.nn.init import normal_, _no_grad_fill_, uniform_

from torch import Tensor
from typing import List, Tuple, Dict, Union

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


@NNRound.register
class VectorQuantizer(nn.Module):
    def __init__(self, num=4096, dim=512, init='normal', reinit_interval=None, threshold=None):
        super().__init__()
        self.num = num
        self.dim = dim
        self.embed = nn.Embedding(num, dim, padding_idx=0)
        if init == 'normal':
            uniform_(self.embed.weight.data, -1.0, 1.0)
        else:
            raise Exception(f'Unknown initialization method {init}')
        self.register_buffer("count", torch.zeros(num))
        self.register_buffer("iter_count", torch.zeros(()))
        self.reinit_interval = reinit_interval
        self.threshold = min(threshold, 1.0) if threshold else 0.1

    def forward(self, x: Tensor, beta: float=0.25, gamma: float=1.0, require_loss: bool=True) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        # x should be in shape [BHW, D]
        x2 = torch.sum(x ** 2, dim=1, keepdim=True) # [BHW, 1]
        z2 = torch.sum(self.embed.weight ** 2, dim=1) # [embed_num,]
        xz = torch.matmul(x, self.embed.weight.t()) # [BHW, embed_num]
        d = x2 + z2 - 2 * xz # [BHW, embed_dim]
        idx = torch.argmin(d, dim=1) # -> [BHW]
        vq_x = self.embed(idx)

        loss_x = F.mse_loss(x, vq_x.detach()) # to learn x, i.e. the encoding result
        loss_embed = F.mse_loss(x.detach(), vq_x) # to learn vq_x, i.e. the codebook
        loss = loss_x * beta + loss_embed * gamma # per dim, i.e. / (B x H x W)

        vq_x = x + (vq_x - x).detach()

        if self.train:
            self.count.index_add_(0, idx, torch.ones_like(idx).float() / (idx.shape[0]))
            self.iter_count += 1
            if self.reinit_interval and torch.sum(self.count) > self.reinit_interval:
                with torch.no_grad():
                    freq_thres = self.reinit_interval / self.num * self.threshold
                    low_freq_ids = (self.count < freq_thres)
                    k = torch.sum(low_freq_ids) // x.shape[0]
                    r = torch.sum(low_freq_ids) % x.shape[0]
                    print(f're-init, count_sum={torch.sum(self.count)}, k={k}, r={r}, num={k * x.shape[0] + r}')
                    x_ = torch.cat((x.repeat(k, 1), x[:r]), dim=0)
                    self.embed.weight.data[low_freq_ids] = x_
                    self.count.fill_(0.0)

        if require_loss:
            return vq_x, loss
        else:
            return vq_x

