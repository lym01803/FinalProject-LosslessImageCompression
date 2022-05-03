import torch
import random
import numpy
import math
from torchvision import utils as vutils

import argparse
import yaml
import os
from tqdm import tqdm

from flows import NNFlows
from vqvae import VQVAE
from moduleregister import Register

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        for line in f.readlines():
            print(line, end='')

    config = yaml.load(open(args.config, 'r'), Loader=yaml.SafeLoader)

    train = config.pop('train')
    train.pop('trainer')
    model_config = train.pop('model')
    model = Register.get(model_config.pop('name'))(**model_config)
    model = model.cuda()
    model.eval()
    
