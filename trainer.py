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

import numpy as np
import os
import math
from tqdm import tqdm
import argparse
from PIL import Image

from moduleregister import Register
from flows import ConditionalFlows, IDFlows, NNFlows
from roundlib import Round
from vqvae import VQVAE, EnDecoder
from extenddim import Patching


def cycle(iterable):
    while True:
        for item in iterable:
            yield item


@Register.register
class CommonDataLoader:
    def __init__(self, path, batch_size, shuffle=True, resize=None, centercrop=None, nbits=8, train=False, pad=None):
        self.path = path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.resize = resize
        self.centercrop = centercrop
        self.transform = transforms.Compose([
            transforms.CenterCrop(tuple(centercrop)),
            transforms.Resize(tuple(resize)),
            transforms.ToTensor()
        ])
        self.dataset = dset.ImageFolder(root=path, transform=self.transform)
        self.loader = DataLoader(self.dataset, batch_size, shuffle=shuffle)
        self.train = train
        if train:
            self.iter = cycle(self.loader)
        else:
            self.iter = iter(self.loader)
        self.round = Round(nbits=nbits)
        self.pad = nn.ReplicationPad2d(padding=(0, pad[1], 0, pad[0])) if pad else None
    
    def __iter__(self):
        return self

    def __next__(self):
        try:
            data, label = next(self.iter)
            if self.pad:
                data = self.pad(data)
            return self.round(data)
        except StopIteration as e:
            self.iter = iter(self.loader)
            raise e


@Register.register
class CustomDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, nbits=8, train=False, pad=None):
        self.dataset = Register.get(dataset.pop('name'))(**dataset)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.loader = DataLoader(self.dataset, batch_size, shuffle=shuffle)
        self.train = train
        if train:
            self.iter = cycle(self.loader)
        else:
            self.iter = iter(self.loader)
        self.round = Round(nbits=nbits)
        self.pad = nn.ReplicationPad2d(padding=(0, pad[1], 0, pad[0])) if pad else None
    
    def __iter__(self):
        return self

    def __next__(self):
        try:
            data = next(self.iter)
            if self.pad:
                data = self.pad(data)
            return self.round(data)
        except StopIteration as e:
            self.iter = iter(self.loader)
            raise e


@Register.register
class ImageNet64Dataset(Dataset):
    def __init__(self, path, size=[3, 64, 64], train=True):
        self.path = path
        self.size = size
        self.train = train
        self.datas = []
        self.lens = []
        if train:
            for i in range(10):
                npz = np.load(os.path.join(path, f'train_data_batch_{i+1}.npz'))
                npz = npz['data']
                self.datas.append(npz)
                self.lens.append(npz.shape[0])
        else:
            npz = np.load(os.path.join(path, f'val_data.npz'))
            npz = npz['data']
            self.datas.append(npz)
            self.lens.append(npz.shape[0])
        self.transform = transforms.Compose([
            transforms.Resize((size[1], size[2])),
            transforms.ToTensor()
        ])

    def toImage(self, nparr):
        x = nparr.reshape(*self.size)
        x = np.transpose(x, (1, 2, 0))
        img = Image.fromarray(x.astype('uint8')).convert('RGB')
        img_tensor = self.transform(img)
        return img_tensor

    def __len__(self):
        return sum(self.lens)

    def __getitem__(self, index):
        data_idx = 0
        while index >= self.lens[data_idx]:
            index -= self.lens[data_idx]
            data_idx += 1
        data = self.datas[data_idx][index]
        return self.toImage(data)


@Register.register
class WarmUpScheduler(LambdaLR):
    def __init__(self, optimizer, warmup, beta):
        self.optimizer = optimizer
        self.warmup = warmup
        self.beta = beta
        def lr_lambda(epoch):
            return min(1., (epoch + 1) / self.warmup) * np.power(self.beta, epoch + 1 - warmup)
        super().__init__(optimizer=optimizer, lr_lambda=lr_lambda)


Register.record['Adamax'] = Adamax
Register.record['Adam'] = Adam
Register.record['SGD'] = SGD

class Trainer:
    def __init__(self,
                 model,
                 train_dataloader,
                 test_dataloader,
                 optimizer,
                 scheduler,
                 max_step,
                 step_per_epoch,
                 evaluate_interval,
                 save_interval,
                 save_path,
                 writer_path):
    
        if 'load_path' in model:
            load_path = model.pop('load_path')
        else:
            load_path = None
        self.model = NNFlows.get(model.pop('name'))(**model).cuda()
        self.trainloader = Register.get(train_dataloader.pop('name'))(**train_dataloader)
        self.testloader = Register.get(test_dataloader.pop('name'))(**test_dataloader)
        self.optimizer = Register.get(optimizer.pop('name'))(self.model.parameters(), **optimizer)
        self.scheduler = Register.get(scheduler.pop('name'))(self.optimizer, **scheduler)
        self.max_step = max_step
        self.step_per_epoch = step_per_epoch
        self.evaluate_interval = evaluate_interval
        self.save_interval = save_interval
        self.save_path = save_path
        self.writer = SummaryWriter(log_dir=writer_path)
        self.step = 0
        if load_path:
            pass

    def train(self):
        for iter in tqdm(range(self.max_step)):
            self.step += 1
            
            data = next(self.trainloader).cuda()
            
            self.model.train()

            x, means, logscales, logv = self.model.forward(data, None)
            logP, logPs = self.model.log_likelihood(x, means, logscales)
            loss = torch.mean(-logP, dim=0)
            loss.backward()

            self.optimizer.step()
            self.model.zero_grad()
            if self.step % self.step_per_epoch == 0:
                self.scheduler.step()

            self.writer.add_scalar('train loss', loss.item(), self.step)
            bpd = loss.item() / math.log(2.)
            self.writer.add_scalar('train bpd', bpd, self.step)

            if (self.step % self.step_per_epoch == 0 and self.step < self.evaluate_interval) or self.step % self.evaluate_interval == 0:
                print()
                for splitid, latent in enumerate(x):
                    max_z = torch.max(latent * 256).item()
                    min_z = torch.min(latent * 256).item()
                    bpd_for_split = torch.mean(-logPs[splitid], dim=0).item() / math.log(2.)
                    print(f'split_id: {splitid} , max_z : {max_z} , min_z : {min_z} , bpd_for_split : {bpd_for_split}')
                print()

                bpds = []
                self.model.eval()
                for data in self.testloader:
                    data = data.cuda()
                    with torch.no_grad():
                        x, means, logscales, logv = self.model.forward(data, None)
                        logP, logPs = self.model.log_likelihood(x, means, logscales)
                        loss = torch.mean(-logP, dim=0)
                        bpd = loss.item() / math.log(2.)
                        bpds.append(bpd)
                self.writer.add_scalar('test bpd', sum(bpds) / len(bpds), self.step)

                self.model.inverse()
                latents = []
                bs = 16
                with torch.no_grad():
                    for shape in self.model.latents_shape:
                        noise = self.model.dist.sample(torch.zeros((bs,)+shape), torch.zeros((bs,)+shape))
                        latents.append(noise.cuda())
                    for t in [0.25, 0.5, 0.75]:
                        generated = self.model.generated_from_noise([latent * t for latent in latents])
                        self.writer.add_image(f't={t}', vutils.make_grid(generated, nrow=4), self.step)

            if (self.step % self.step_per_epoch == 0 and self.step < self.save_interval) or self.step % self.save_interval == 0:
                state = dict(
                    model=self.model.state_dict(),
                    optimizer=self.optimizer.state_dict(),
                    step=self.step
                )
                torch.save(state, self.save_path)


@Register.register
class TwoLevelTrainer:
    def __init__(self,
                 model,
                 train_dataloader,
                 test_dataloader,
                 optimizer,
                 scheduler,
                 max_step,
                 step_per_epoch,
                 evaluate_interval,
                 save_interval,
                 save_path,
                 writer_path):
    
        if 'load_path' in model:
            load_path = model.pop('load_path')
        else:
            load_path = None
        self.model = NNFlows.get(model.pop('name'))(**model).cuda()
        self.trainloader = Register.get(train_dataloader.pop('name'))(**train_dataloader)
        self.testloader = Register.get(test_dataloader.pop('name'))(**test_dataloader)
        self.optimizer = Register.get(optimizer.pop('name'))(self.model.parameters(), **optimizer)
        self.scheduler = Register.get(scheduler.pop('name'))(self.optimizer, **scheduler)
        self.max_step = max_step
        self.step_per_epoch = step_per_epoch
        self.evaluate_interval = evaluate_interval
        self.save_interval = save_interval
        self.save_path = save_path
        self.writer = SummaryWriter(log_dir=writer_path)
        self.step = 0
        self.dist = DLogistic()
        if load_path:
            pass

    def train(self):
        for iter in tqdm(range(self.max_step)):
            self.step += 1
            
            data = next(self.trainloader).cuda()
            
            self.model.train()

            x, means, logscales, bpd, bpd1, bpd2, logv = self.model.forward(data, None)

            self.optimizer.step()
            self.model.zero_grad()
            if self.step % self.step_per_epoch == 0:
                self.scheduler.step()

            # self.writer.add_scalar('train loss', loss.item(), self.step)
            # bpd = loss.item() / math.log(2.)
            self.writer.add_scalar('train bpd', bpd, self.step)
            self.writer.add_scalar('train bpd 1', bpd1, self.step)
            self.writer.add_scalar('train bpd 2', bpd2, self.step)

            if self.step <= 3 or (self.step % self.step_per_epoch == 0 and self.step < self.evaluate_interval) or self.step % self.evaluate_interval == 0:
                print()
                for splitid, latent in enumerate(x):
                    max_z = torch.max(latent * 256).item()
                    min_z = torch.min(latent * 256).item()
                    print(f'part_id: {splitid} , max_z : {max_z} , min_z : {min_z}')
                print()

                bpds = []
                bpd1s = []
                bpd2s = []
                self.model.eval()
                for data in tqdm(self.testloader):
                    data = data.cuda()
                    with torch.no_grad():
                        x, means, logscales, bpd, bpd1, bpd2, logv = self.model.forward(data, None, train=False)
                        bpds.append(bpd)
                        bpd1s.append(bpd1)
                        bpd2s.append(bpd2)

                self.writer.add_scalar('test bpd', sum(bpds) / len(bpds), self.step)
                self.writer.add_scalar('test bpd 1', sum(bpd1s) / len(bpd1s), self.step)
                self.writer.add_scalar('test bpd 2', sum(bpd2s) / len(bpd2s), self.step)

                self.model.inverse()
                latents = []
                bs = 4
                with torch.no_grad():
                    for shape in self.model.latents_shape:
                        noise = self.dist.sample(torch.zeros((bs,)+shape), torch.zeros((bs,)+shape))
                        latents.append(noise.cuda())
                    for t in [0.25, 0.5, 0.75, 1.0]:
                        generated = self.model.generated_from_noise([latent * t for latent in latents])
                        self.writer.add_image(f't={t}', vutils.make_grid(generated, nrow=4), self.step)

            if (self.step % self.step_per_epoch == 0 and self.step < self.save_interval) or self.step % self.save_interval == 0:
                state = dict(
                    model=self.model.state_dict(),
                    optimizer=self.optimizer.state_dict(),
                    step=self.step
                )
                torch.save(state, self.save_path)


@Register.register
class VQVAETrainer:
    def __init__(self,
                 model,
                 train_dataloader,
                 test_dataloader,
                 optimizer,
                 scheduler,
                 max_step,
                 step_per_epoch,
                 evaluate_interval,
                 save_interval,
                 save_path,
                 writer_path,
                 train_args={}):
    
        if 'load_path' in model:
            load_path = model.pop('load_path')
        else:
            load_path = None
        self.model = EnDecoder.get(model.pop('name'))(**model).cuda()
        self.trainloader = Register.get(train_dataloader.pop('name'))(**train_dataloader)
        self.testloader = Register.get(test_dataloader.pop('name'))(**test_dataloader)
        self.optimizer = Register.get(optimizer.pop('name'))(self.model.parameters(), **optimizer)
        self.scheduler = Register.get(scheduler.pop('name'))(self.optimizer, **scheduler)
        self.max_step = max_step
        self.step_per_epoch = step_per_epoch
        self.evaluate_interval = evaluate_interval
        self.save_interval = save_interval
        self.save_path = save_path
        self.writer = SummaryWriter(log_dir=writer_path)
        self.step = 0
        self.alpha = train_args.pop('alpha')
        self.train_args = train_args
        if load_path:
            pass

    def train(self):
        for iter in tqdm(range(self.max_step)):
            self.step += 1
            
            data = next(self.trainloader).cuda()
            
            self.model.train()
            # the input should be in [-1, 1], but data is default in [0, 1]
            x, vqloss = self.model.encode(x=(data - 0.5) / 0.5, require_loss=True, **self.train_args)
            x = self.model.decode(x) * 0.5 + 0.5 # x to [0, 1]
            recloss = - self.model.dist.log_prob(x=data, y=x) # x is the param, data is the sample
            recloss = torch.mean(recloss)
            loss = self.alpha * recloss + vqloss
            loss.backward()

            self.optimizer.step()
            self.model.zero_grad()
            if self.step % self.step_per_epoch == 0:
                self.scheduler.step()

            self.writer.add_scalar('train loss', loss.item(), self.step)
            self.writer.add_scalar('train recloss', recloss.item(), self.step)
            self.writer.add_scalar('train vqloss', vqloss.item(), self.step)

            bpd = recloss.item() / math.log(2.)
            self.writer.add_scalar('train bpd', bpd, self.step)

            if (self.step % self.step_per_epoch == 0 and self.step < self.evaluate_interval) or self.step % self.evaluate_interval == 0:
                bpds = []
                self.model.eval()
                for data in self.testloader:
                    data = data.cuda()
                    with torch.no_grad():
                        x = self.model.encode(x=(data - 0.5) / 0.5, require_loss=False)
                        x = self.model.decode(x) * 0.5 + 0.5
                        recloss = - self.model.dist.log_prob(x=data, y=x) # x is the param, data is the sample
                        recloss = torch.mean(recloss)
                        bpd = recloss.item() / math.log(2.)
                        bpds.append(bpd)
                self.writer.add_scalar('test bpd', sum(bpds) / len(bpds), self.step)
                self.writer.add_image(f're-construct', vutils.make_grid(x, nrow=4), self.step)


            if (self.step % self.step_per_epoch == 0 and self.step < self.save_interval) or self.step % self.save_interval == 0:
                state = dict(
                    model=self.model.state_dict(),
                    optimizer=self.optimizer.state_dict(),
                    step=self.step
                )
                torch.save(state, self.save_path)


@Register.register
class ResidualTrainer:
    def __init__(self,
                 flows,
                 vqvae, # require checkpoint for vqvae
                 input_size,
                 train_dataloader,
                 test_dataloader,
                 patch_batch_size,
                 optimizer,
                 scheduler,
                 max_step,
                 step_per_epoch,
                 evaluate_interval,
                 save_interval,
                 save_path,
                 writer_path):

        self.model = NNFlows.get(flows.pop('name'))(**flows).cuda()
        self.model: IDFlows

        self.vqvae_path = vqvae.pop('checkpoint')
        self.vqvae = EnDecoder.get(vqvae.pop('name'))(**vqvae)
        self.vqvae.load_state_dict(torch.load(self.vqvae_path)['model'])
        self.vqvae = self.vqvae.cuda()
        self.vqvae.eval() # This state will not be changed. The vqvae is only used to inference.
        self.vqvae: VQVAE
        
        self.trainloader = Register.get(train_dataloader.pop('name'))(**train_dataloader)
        self.testloader = Register.get(test_dataloader.pop('name'))(**test_dataloader)
        self.optimizer = Register.get(optimizer.pop('name'))(self.model.parameters(), **optimizer)
        self.scheduler = Register.get(scheduler.pop('name'))(self.optimizer, **scheduler)
        self.max_step = max_step
        self.step_per_epoch = step_per_epoch
        self.evaluate_interval = evaluate_interval
        self.save_interval = save_interval
        self.save_path = save_path
        self.writer = SummaryWriter(log_dir=writer_path)
        self.step = 0
        self.input_size = input_size
        self.patch = Patching(input_size[0], input_size[1], self.model.H, self.model.W)
        self.patch_batch_size = patch_batch_size

    def train(self):
        for iter in tqdm(range(self.max_step)):
            self.vqvae.eval()
            self.step += 1
            
            data = next(self.trainloader).cuda() # The original paded data
            with torch.no_grad():
                rec = self.vqvae.forward((data - 0.5) / 0.5, require_loss=False) * 0.5 + 0.5
                rec = self.model.round(rec) # should be rounded, because res = data - rec should be integer
                res = data - rec
                patches, _ = self.patch(res, None)
                if isinstance(self.model, ConditionalFlows):
                    rec_patches, _ = self.patch(rec, None)
            # patches will be in shape (B * (H // h) * (W // w), 3, h, w)
            # So B should not be too large, B = 2 or 4 will be OK
            
            self.model.train()
            if self.patch_batch_size == 0:
                if isinstance(self.model, ConditionalFlows):
                    x, means, logscales, logv = self.model.forward(patches, None, rec_patches)
                else:
                    x, means, logscales, logv = self.model.forward(patches, None)
                logP, logPs = self.model.log_likelihood(x, means, logscales)
                loss = torch.mean(-logP, dim=0)
                loss.backward()

                self.optimizer.step()
                self.model.zero_grad()
                
                self.writer.add_scalar('train loss', loss.item(), self.step)
                bpd = loss.item() / math.log(2.)
                self.writer.add_scalar('train bpd', bpd, self.step)
            
            else:
                bpds = []
                loss_list = []
                bs = self.patch_batch_size
                minibatch_num = (patches.shape[0] + bs - 1) // bs
                perm = torch.randperm(patches.shape[0])
                shuffled_patches = patches[perm]
                shuffled_rec_patches = rec_patches[perm]
                for i in range(minibatch_num):
                    if isinstance(self.model, ConditionalFlows):
                        x, means, logscales, logv = self.model.forward(shuffled_patches[i*bs:i*bs+bs], 
                                                                       None, 
                                                                       shuffled_rec_patches[i*bs:i*bs+bs])
                    else:
                        x, means, logscales, logv = self.model.forward(shuffled_patches[i*bs:i*bs+bs], None)
                    logP, logPs = self.model.log_likelihood(x, means, logscales)
                    loss = torch.mean(-logP, dim=0)
                    loss.backward()

                    self.optimizer.step()
                    self.model.zero_grad()

                    loss_list.append(loss.item())
                    bpds.append(loss.item() / math.log(2.))

                if len(bpds) > 0:
                    self.writer.add_scalar('train loss', sum(loss_list) / len(loss_list), self.step)
                    self.writer.add_scalar('train bpd', sum(bpds) / len(bpds), self.step)


            if self.step % self.step_per_epoch == 0:
                self.scheduler.step()
            

            if (self.step % self.step_per_epoch == 0 and self.step < self.evaluate_interval) or self.step % self.evaluate_interval == 0:
                print()
                for splitid, latent in enumerate(x):
                    max_z = torch.max(latent * (2 ** self.model.round.nbits)).item()
                    min_z = torch.min(latent * (2 ** self.model.round.nbits)).item()
                    bpd_for_split = torch.mean(-logPs[splitid], dim=0).item() / math.log(2.)
                    print(f'split_id: {splitid} , max_z : {max_z} , min_z : {min_z} , bpd_for_split : {bpd_for_split}')
                print()

                bpds = []
                self.model.eval()
                for data in tqdm(self.testloader):
                    data = data.cuda()
                    with torch.no_grad():
                        rec = self.vqvae.forward((data - 0.5) / 0.5, require_loss=False) * 0.5 + 0.5
                        rec = self.model.round(rec) # should be rounded, because res = data - rec should be integer
                        res = data - rec
                        patches, _ = self.patch(res, None)
                        if isinstance(self.model, ConditionalFlows):
                            x, means, logscales, logv = self.model.forward(patches,
                                                                           None,
                                                                           self.patch(rec, None)[0])
                        else:
                            x, means, logscales, logv = self.model.forward(patches, None)
                        logP, logPs = self.model.log_likelihood(x, means, logscales)
                        loss = torch.mean(-logP, dim=0)
                        bpd = loss.item() / math.log(2.)
                        bpds.append(bpd)
                self.writer.add_scalar('test bpd', sum(bpds) / len(bpds), self.step)

                # show the original image, original res, rec, res and rec + res
                self.writer.add_image('original image', vutils.make_grid(data, nrow=4), self.step)
                self.writer.add_image('rec by vqvae', vutils.make_grid(rec, nrow=4), self.step)
                self.writer.add_image('residual (shifted by 0.5)', vutils.make_grid(res + 0.5, nrow=4), self.step)
                with torch.no_grad():
                    generated_res = self.model.generated_from_latents(x)
                    generated_res = self.patch.backward(generated_res) # in shape [B, 3, H, W]
                    rec_image = rec + generated_res
                self.writer.add_image('decode residual (shifted by 0.5)', vutils.make_grid(generated_res + 0.5, nrow=4), self.step)
                self.writer.add_image('rec image', vutils.make_grid(rec_image, nrow=4), self.step)
                self.writer.add_scalar('test rec error', torch.norm(data - rec_image), self.step)
                

            if (self.step % self.step_per_epoch == 0 and self.step < self.save_interval) or self.step % self.save_interval == 0:
                state = dict(
                    model=self.model.state_dict(),
                    optimizer=self.optimizer.state_dict(),
                    step=self.step
                )
                torch.save(state, self.save_path)

