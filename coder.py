import torch
from torch import nn

from distlib import DLogistic

import numpy as np
import os
import math
from tqdm import tqdm
import argparse
from PIL import Image
import time
import random

from rans.rans import encode, decode


def Encode(latents, means, logscales, x=(1<<32)):
    buffers = []
    for i in range(len(latents)):
        idx = i
        latent = latents[idx].reshape(-1).tolist()
        scale = torch.exp(logscales[idx]).reshape(-1).tolist()
        mean = means[idx].reshape(-1).tolist()
        x, buf = encode(x, len(latent), latent, mean, scale)
        buffers.append(buf)
    return x, buffers

def Decode(buffers, means, logscales, x):
    # buffers = buffers[::-1]
    latents = []
    for i in range(len(means)):
        idx = len(means) - 1 - i
        mean = means[idx].reshape(-1).tolist()
        scale = torch.exp(logscales[idx]).reshape(-1).tolist()
        x, latent = decode(x, buffers[idx][::-1], len(mean), mean[::-1], scale[::-1])
        latents.append(torch.tensor(latent[::-1]).to(means[idx]).reshape(*(means[idx].shape)))
    return x, latents[::-1]


if __name__ == '__main__':

    n = 500000

    mean = [random.randint(-32, 32) / 256 for i in range(n)]
    scale = [math.exp(random.random() * 0.01 - 0.005) for i in range(n)]
    msg = [round((mean[i] + scale[i] * (1. * random.random() - .5)) * 256) / 256 for i in range(n)]
    # print([m * 256 for m in mean], [m * 256 for m in msg])
    x = (1 << 32)

    print('start')
    t1 = time.time()
    x, buf = encode(x, n, msg, mean, scale)
    t2 =time.time()
    print(x, len(buf), buf[0], min(msg) * 256, max(msg) * 256, mean[:10], scale[:10], msg[:10])
    mean_ = mean[::-1]
    scale_ = scale[::-1]
    t3 = time.time()
    x, msg_rec = decode(x, buf[::-1], n, mean_, scale_)
    t4 = time.time()
    msg_rec = msg_rec[::-1]

    print(x, 1 << 32)
    print(f'encode: {t2-t1}', f'decode: {t4-t3}')
    # print('decode: ', x, msg_rec)
    # print('original: ', 1 << 32, msg)

    count = 0
    for i in range(n):
        if (msg[i] - msg_rec[i]) > 1e-6:
            print(i, msg[i], msg_rec[i])
            count += 1
    print(count)
