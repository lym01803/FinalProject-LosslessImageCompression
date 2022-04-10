import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import normal_, _no_grad_fill_
from torch.nn.parameter import Parameter

import numpy as np
import math
import os
import sys
import random

import moduleregister

class ActivateFunc(moduleregister.Register):
    def __init__(self):
        super().__init__()

