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
from moduleregister import Register


def Sample(model, shape, output, batch_size=16, nums=16, nrow=8, temp=[0.25, 0.5, 0.75]):
    noises = []
    batch_num = nums // batch_size + (1 if nums % batch_size else 0)
    for i in range(batch_num):
        noise = []
        start = i * batch_size
        width = min(start + batch_size, nums) - start
        for sh in shape:
            noise.append(model.dist.sample(
                torch.zeros([width] + sh),
                torch.zeros([width] + sh)
            ))
        noises.append(noise)
    model.eval()
    with torch.no_grad():
        for t in tqdm(temp):
            images = []
            for i in range(batch_num):
                noise = [(t * z).cuda() for z in noises[i]]
                generated = model.generated_from_noise(noise)
                generated = generated.cpu()
                images.append(generated)
            image = torch.cat(images, dim=0)
            name = os.path.join(output, f'sample_t_{t:0<4.2f}.png')
            vutils.save_image(image, name, nrow=nrow)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, required=True)

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)
    
    model_config = config.pop('model')
    options = config.pop('options')

    chkpt = model_config.pop('checkpoint')

    model = NNFlows.get(model_config.pop('name'))(**model_config)
    model.load_state_dict(torch.load(chkpt)['model'])
    model = model.cuda()
    model.eval()

    output = options.pop('output')
    os.makedirs(output, exist_ok=True)

    shape = options.pop('latent_shape')
    if 'sample' in options:
        sample_options = options.pop('sample')
        Sample(model=model, shape=shape, output=output, **sample_options)
    
