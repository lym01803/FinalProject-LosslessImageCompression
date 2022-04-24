import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import random
from copy import deepcopy

import math

from moduleregister import Register
from roundlib import NNRound, Round, VectorQuantizer
from distlib import NNDistribution
from priorlib import NNPrior
from nnblock import NNBlock


class EnDecoder(Register):
    def __init__(self):
        super().__init__()


@Register.register
class VQEncoder(nn.Module):
    def __init__(self, in_channel, out_channel, block, block_num, hidden_dims=[128, 256], batch_norm=False):
        super().__init__()
        self.blocks = nn.ModuleList()
        ch = in_channel
        # The image size will be reduced to half for each hidden_dim.
        # So if the hidden_dims list have the length of 2, the size H x W will be 
        # reduced to (H // 4) x (W // 4)
        for dim in hidden_dims:
            if batch_norm:
                self.blocks.append(nn.Sequential(
                    nn.Conv2d(ch, dim, 4, 2, 1),
                    nn.LeakyReLU(inplace=True),
                    nn.BatchNorm2d(dim)
                ))
            else:
                self.blocks.append(nn.Sequential(
                    nn.Conv2d(ch, dim, 4, 2, 1),
                    nn.LeakyReLU(inplace=True)
                ))
            ch = dim
        if batch_norm:
            self.blocks.append(nn.Sequential(
                nn.Conv2d(ch, ch, 3, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.BatchNorm2d(ch)
            ))
        else:
            self.blocks.append(nn.Sequential(
                nn.Conv2d(ch, ch, 3, 1, 1),
                nn.LeakyReLU(inplace=True)
            ))
        block_type = NNBlock.get(block.pop('name'))
        for i in range(block_num):
            self.blocks.append(block_type(channel=ch, **deepcopy(block)))
        self.blocks.append(nn.Conv2d(ch, out_channel, 1))
        self.act = nn.Tanh()

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.act(x)
        return x 


@Register.register
class VQDecoder(nn.Module):
    def __init__(self, in_channel, out_channel, block, block_num, hidden_dims=[256, 128], batch_norm=False):
        super().__init__()
        self.blocks = nn.ModuleList()
        ch = hidden_dims[0]
        if batch_norm:
            self.blocks.append(nn.Sequential(
                nn.Conv2d(in_channel, ch, 1),
                nn.LeakyReLU(inplace=True),
                nn.BatchNorm2d(ch)
            ))
        else:
            self.blocks.append(nn.Sequential(
                nn.Conv2d(in_channel, ch, 1),
                nn.LeakyReLU(inplace=True)
            ))
        block_type = NNBlock.get(block.pop('name'))
        for i in range(block_num):
            self.blocks.append(block_type(channel=ch, **deepcopy(block)))
        self.blocks.append(nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.LeakyReLU(inplace=True)
        ))
        for dim in hidden_dims[1:]:
            if batch_norm:
                self.blocks.append(nn.Sequential(
                    nn.ConvTranspose2d(ch, dim, 4, 2, 1),
                    nn.LeakyReLU(inplace=True),
                    nn.BatchNorm2d(dim)
                ))
            else:
                self.blocks.append(nn.Sequential(
                    nn.ConvTranspose2d(ch, dim, 4, 2, 1),
                    nn.LeakyReLU(inplace=True)
                ))
            ch = dim 
        self.blocks.append(nn.Sequential(
            nn.ConvTranspose2d(ch, out_channel, 4, 2, 1),
            nn.Tanh()
        ))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


@EnDecoder.register
class VQVAE(nn.Module):
    def __init__(self, channel, embed_num, embed_dim, encoder, decoder, distribution, vectorquantizer={}, hidden_dims=[128, 256], batch_norm=False):
        super().__init__()
        self.channel = channel 
        self.embed_num = embed_num
        self.embed_dim = embed_dim
        self.encoder = Register.get(encoder.pop('name'))(
            in_channel=channel,
            out_channel=embed_dim,
            hidden_dims=hidden_dims,
            batch_norm=batch_norm,
            **encoder
        )
        self.decoder = Register.get(decoder.pop('name'))(
            in_channel=embed_dim,
            out_channel=channel,
            hidden_dims=hidden_dims[::-1],
            batch_norm=batch_norm,
            **decoder
        )
        self.vq = VectorQuantizer(num=embed_num, dim=embed_dim, **vectorquantizer)
        self.dist = NNDistribution.get(distribution.pop('name'))(**distribution)

    def encode(self, x, beta=0.25, gamma=1.0, require_loss=True):
        x = self.encoder(x)
        x = x.permute(0, 2, 3, 1).contiguous() # B x C x H x W -> B x H x W x C
        shape = x.shape
        x = x.view(-1, shape[-1])
        if require_loss:
            vq_x, loss = self.vq.forward(x, beta=beta, gamma=gamma, require_loss=require_loss)
            vq_x = vq_x.view(shape)
            vq_x = vq_x.permute(0, 3, 1, 2).contiguous() # B x H x W x C -> B x C x H x W
            return vq_x, loss
        else:
            vq_x = self.vq.forward(x, beta=beta, gamma=gamma, require_loss=require_loss)
            vq_x = vq_x.view(shape)
            vq_x = vq_x.permute(0, 3, 1, 2).contiguous() # B x H x W x C -> B x C x H x W
            return vq_x
        
    def decode(self, x):
        x = self.decoder(x)
        return x

    def forward(self, x, beta=0.25, gamma=1.0, require_loss=True):
        if require_loss:
            vq_x, vqloss = self.encode(x, beta=beta, gamma=gamma, require_loss=require_loss)
            output = self.decode(vq_x)
            return output, vqloss 
        else:
            vq_x = self.encode(x, beta=beta, gamma=gamma, require_loss=require_loss)
            output = self.decode(vq_x)
            return output 

