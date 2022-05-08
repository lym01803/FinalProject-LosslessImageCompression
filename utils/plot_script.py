import torch
import numpy as np
import pandas as pd
import os
import sys
import random
from tqdm import tqdm 
from matplotlib import pyplot as plt
import seaborn as sns

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def draw(L, th=10000):
    df = pd.DataFrame()
    df['step'] = [l[0] for l in L][:th]
    df['value'] = [l[1] for l in L][:th]
    sns.set_theme(style="darkgrid")
    sns.lineplot(data=df, x='step', y='value')
    plt.savefig('./fig/train_bpd.png')

if __name__ == '__main__':
    log = '/home/yanming/files/IDFLOW/logs/log_for_vqvae_for_celeba_full_reinit/'
    event = EventAccumulator(log)
    event.Reload()
    print(event.Tags()['scalars'])
    train_bpd = event.Scalars('train bpd')
    test_bpd = event.Scalars('test bpd')
    vqloss = event.Scalars('train vqloss')

    f = lambda x: list(map(lambda v: [v.step, v.value], x))

    train_bpd = f(train_bpd)
    test_bpd = f(test_bpd)
    vqloss = f(vqloss)

    # print(train_bpd)
    draw(train_bpd, th=100)
    
    