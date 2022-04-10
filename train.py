from sched import scheduler
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import transforms
from torch.optim import SGD, Adam, Adamax
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dset
import torchvision.utils as vutils
from distlib import DLogistic
from flows import NNFlows
from trainer import Trainer

import numpy as np
import os
import math
from tqdm import tqdm
import argparse
import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        for line in f.readlines():
            print(line, end='')

    config = yaml.load(open(args.config, 'r'), Loader=yaml.SafeLoader)
    
    main_trainer = Trainer(**config.pop('train'))
    main_trainer.train()

