import torch
import random
import numpy
import math
from torchvision import utils as vutils
from torchvision import transforms

from PIL import Image

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


def Interpolate(model, folder, images, shape, output, nrow=8):
    transform = transforms.Compose([
        transforms.CenterCrop(shape),
        transforms.Resize(shape),
        transforms.ToTensor()
    ])
    ids = [0, nrow-1, (nrow-1)*nrow, nrow*nrow-1]
    imgs_tensors = torch.zeros((nrow*nrow, 3, shape[0], shape[1])).cuda()
    la = []

    for i, image in enumerate(images):
        path = os.path.join(folder, image)
        img = Image.open(path)
        img = transform(img).cuda().unsqueeze(dim=0)
        img = torch.round(img * 256) / 256
        imgs_tensors[ids[i]] = img 
        with torch.no_grad():
            latents, ms_, logs_, logv = model.forward(x=img, logv=None)
        for j, latent, in enumerate(latents):
            if len(la) <= j:
                la.append(torch.zeros((nrow*nrow,)+latent.shape[1:]).cuda())
            la[j][ids[i]] = (latent[0] - ms_[j][0]) / torch.exp(logs_[j][0])
    
    for i in range(8):
        for j in range(8):
            a = j / 7
            b = i / 7
            coef = [
                (1 - a) * (1 - b),
                a * (1 - b),
                (1 - a) * b,
                a * b
            ]
            id = i * 8 + j
            if id not in ids:
                for latent in la:
                    # norm_ = torch.tensor((0.)).cuda()
                    for k in range(4):
                        latent[id] += coef[k] * latent[ids[k]]
                        # norm_ += coef[k] * torch.norm(latent[ids[k]])
                    # latent[id] = latent[id] / torch.norm(latent[id]) * norm_
                for k in range(4):
                    imgs_tensors[id] += coef[k] * imgs_tensors[ids[k]]

    imgs_tensors = torch.round(imgs_tensors * 256) / 256
    vutils.save_image(imgs_tensors, os.path.join(output, 'hard-linear.png'), nrow=nrow)
    with torch.no_grad():
        # for lala in la:
            # print(lala.shape)
        imgs_tensors = model.generated_from_noise(la)
    vutils.save_image(imgs_tensors, os.path.join(output, 'interpolate.png'), nrow=nrow)

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

    if 'latent_shape' in options:
        shape = options.pop('latent_shape')
    if 'sample' in options:
        sample_options = options.pop('sample')
        Sample(model=model, shape=shape, output=output, **sample_options)
    if 'interpolate' in options:
        inter_options = options.pop('interpolate')
        Interpolate(model=model, output=output, **inter_options)
    
