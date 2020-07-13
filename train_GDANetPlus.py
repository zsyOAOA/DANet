#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-01-10 22:41:49

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as uData
from math import ceil
from networks import UNetD, UNetG, sample_generator
from datasets.DenoisingDatasets import BenchmarkTrain, BenchmarkTest, FakeTrain, PolyuTrain
from utils import batch_PSNR, batch_SSIM
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
import shutil
import warnings
from pathlib import Path
import commentjson as json

# filter warnings
warnings.simplefilter('ignore', Warning, lineno=0)

_C = 3
_modes = ['train', 'val']

def train_model(net, netG, datasets, optimizer, lr_scheduler, args):
    batch_size = {'train':args['batch_size'], 'val':4}
    data_loader = {phase:uData.DataLoader(datasets[phase], batch_size=batch_size[phase],
                shuffle=True, num_workers=args['num_workers'], pin_memory=True) for phase in _modes}
    num_data = {phase:len(datasets[phase]) for phase in _modes}
    num_iter_epoch = {phase: ceil(num_data[phase] / batch_size[phase]) for phase in _modes}
    step = args['step'] if args['resume'] else 0
    step_img = args['step_img'] if args['resume'] else {x:0 for x in _modes}
    writer = SummaryWriter(str(Path(args['log_dir'])))
    clip_grad = args['clip_normD']
    for epoch in range(args['epoch_start'], args['epochs']):
        mae_per_epoch = {x:0 for x in _modes}
        tic = time.time()
        # train stage
        net.train()
        lr = optimizer.param_groups[0]['lr']
        grad_mean = 0
        phase = 'train'
        for ii, data in enumerate(data_loader[phase]):
            im_noisy_real, im_gt, mask = [x.cuda() for x in data]
            with torch.autograd.no_grad():
                im_noisy_fake = sample_generator(netG, im_gt)
                im_noisy_fake.clamp_(0.0, 1.0)
            im_noisy = im_noisy_real*mask + im_noisy_fake*(1.0-mask)
            optimizer.zero_grad()
            im_denoise = im_noisy - net(im_noisy)
            loss = F.l1_loss(im_denoise, im_gt, reduction='sum') / im_gt.shape[0]

            # backpropagation
            loss.backward()
            # clip the grad
            total_grad = nn.utils.clip_grad_norm_(net.parameters(), clip_grad)
            grad_mean = grad_mean*ii/(ii+1) + total_grad/(ii+1)
            optimizer.step()

            scale = im_noisy.numel() / im_gt.shape[0]
            mae_iter = loss.item() / scale
            mae_per_epoch[phase] += mae_iter
            if (ii+1) % args['print_freq'] == 0:
                template = '[Epoch:{:>2d}/{:<2d}] {:s}:{:0>5d}/{:0>5d}, Loss={:5.2e}, ' + \
                                                                     'Grad:{:.2e}/{:.2e}, lr={:.2e}'
                print(template.format(epoch+1, args['epochs'], phase, ii+1, num_iter_epoch[phase],
                                                               mae_iter, clip_grad, total_grad, lr))
                writer.add_scalar('Train Loss Iter', mae_iter, step)
                step += 1
            if (ii+1) % (20*args['print_freq']) == 0:
                x1 = vutils.make_grid(im_denoise, normalize=True, scale_each=True)
                writer.add_image(phase+' Denoised images', x1, step_img[phase])
                x2 = vutils.make_grid(im_gt, normalize=True, scale_each=True)
                writer.add_image(phase+' GroundTruth', x2, step_img[phase])
                x3 = vutils.make_grid(im_noisy, normalize=True, scale_each=True)
                writer.add_image(phase+' Noisy Image', x3, step_img[phase])
                step_img[phase] += 1

        mae_per_epoch[phase] /= (ii+1)
        clip_grad = min(grad_mean, clip_grad)
        print('{:s}: Loss={:+.2e}, grad_mean={:.2e}'.format(phase, mae_per_epoch[phase], grad_mean))
        print('-'*100)

        # test stage
        net.eval()
        psnr_per_epoch = ssim_per_epoch = 0
        phase = 'val'
        for ii, data in enumerate(data_loader[phase]):
            im_noisy, im_gt = [x.cuda() for x in data]
            with torch.set_grad_enabled(False):
                im_denoise = im_noisy - net(im_noisy)

            im_denoise.clamp_(0.0, 1.0)
            mae_iter = F.l1_loss(im_denoise, im_gt)
            mae_per_epoch[phase] += mae_iter
            psnr_iter = batch_PSNR(im_denoise, im_gt)
            psnr_per_epoch += psnr_iter
            ssim_iter = batch_SSIM(im_denoise, im_gt)
            ssim_per_epoch += ssim_iter
            # print statistics every log_interval mini_batches
            if (ii+1) % 50 == 0:
                log_str = '[Epoch:{:>2d}/{:<2d}] {:s}:{:0>3d}/{:0>3d}, mae={:.2e}, ' + \
                                                                    'psnr={:4.2f}, ssim={:5.4f}'
                print(log_str.format(epoch+1, args['epochs'], phase, ii+1, num_iter_epoch[phase],
                                                                    mae_iter, psnr_iter, ssim_iter))
                # tensorboard summary
                x1 = vutils.make_grid(im_denoise, normalize=True, scale_each=True)
                writer.add_image(phase+' Denoised images', x1, step_img[phase])
                x2 = vutils.make_grid(im_gt, normalize=True, scale_each=True)
                writer.add_image(phase+' GroundTruth', x2, step_img[phase])
                x5 = vutils.make_grid(im_noisy, normalize=True, scale_each=True)
                writer.add_image(phase+' Noisy Image', x5, step_img[phase])
                step_img[phase] += 1

        psnr_per_epoch /= (ii+1)
        ssim_per_epoch /= (ii+1)
        mae_per_epoch[phase] /= (ii+1)
        print('{:s}: mse={:.3e}, PSNR={:4.2f}, SSIM={:5.4f}'.format(phase, mae_per_epoch[phase],
                                                                    psnr_per_epoch, ssim_per_epoch))
        print('-'*100)

        # adjust the learning rate
        lr_scheduler.step()
        # save model
        save_path_model = str(Path(args['model_dir']) / ('model_'+str(epoch+1)))
        torch.save({
            'epoch': epoch+1,
            'step': step+1,
            'step_img': {x:step_img[x]+1 for x in _modes},
            'clip_grad': clip_grad,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict()
            }, save_path_model)
        save_path_model = str(Path(args['model_dir']) / ('model_state_'+str(epoch+1)+'.pt'))
        torch.save(net.state_dict(), save_path_model)

        writer.add_scalars('MAE_epoch', mae_per_epoch, epoch)
        writer.add_scalar('Val PSNR epoch', psnr_per_epoch, epoch)
        writer.add_scalar('Val SSIM epoch', ssim_per_epoch, epoch)
        toc = time.time()
        print('This epoch take time {:.2f}'.format(toc-tic))
    writer.close()
    print('Reach the maximal epochs! Finish training')

def main():
    # set parameters
    with open('./configs/GDANetPlus.json', 'r') as f:
        args = json.load(f)

    # set the available GPUs
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args['gpu_id'])

    # build the network
    net = UNetD(_C, wf=args['wf'], depth=args['depth']).cuda()
    netG = UNetG(_C, wf=args['wf'], depth=args['depth']).cuda()
    # load the generator
    netG.load_state_dict(torch.load(args['pretrain'], map_location='cpu')['G'])

    # optimizer
    optimizer = optim.Adam(net.parameters(), lr=args['lr_D'])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args['milestones'], gamma=0.5)

    if args['resume']:
        if Path(args['resume']).is_file():
            print('=> Loading checkpoint {:s}'.format(str(Path(args['resume']))))
            checkpoint = torch.load(args['resume'])
            args['epoch_start'] = checkpoint['epoch']
            args['step'] = checkpoint['step']
            args['step_img'] = checkpoint['step_img']
            args['clip_normD'] = checkpoint['clip_grad']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            net.load_state_dict(checkpoint['model_state_dict'])
            print('=> Loaded checkpoint {:s} (epoch {:d})'.format(args['resume'], checkpoint['epoch']))
        else:
            sys.exit('Please provide corrected model path!')
    else:
        args['epoch_start'] = 0
        if Path(args['log_dir']).is_dir():
            shutil.rmtree(args['log_dir'])
        Path(args['log_dir']).mkdir()
        if Path(args['model_dir']).is_dir():
            shutil.rmtree(args['model_dir'])
        Path(args['model_dir']).mkdir()

    for key, value in args.items():
        print('{:<15s}: {:s}'.format(key,  str(value)))

    # making dataset
    num_iters_sidd = 5000
    num_iters_renoir = 2000
    num_iters_poly = 1000
    num_iters_fake = ceil((num_iters_sidd+num_iters_renoir+num_iters_poly) * args['fake_ratio'])
    path_list_Poly = sorted([str(x) for x in Path(args['Poly_dir']).glob('*_real.JPG')])
    print('Number of images in Poly Dataset: {:d}'.format(len(path_list_Poly)))
    path_list_fake = sorted([str(x) for x in Path(args['fake_dir']).glob('*/*.jpg')])
    print('Number of images in fake floder: {:d}'.format(len(path_list_fake)))
    datasets_list = [BenchmarkTrain(h5_file=args['SIDD_train_h5'],
                                    length=num_iters_sidd*args['batch_size'],
                                    pch_size=args['patch_size'],
                                    mask=True),
                     BenchmarkTrain(h5_file=args['Renoir_train_h5'],
                                    length=num_iters_renoir*args['batch_size'],
                                    pch_size=args['patch_size'],
                                    mask=True),
                     PolyuTrain(path_list=path_list_Poly,
                                length=num_iters_poly*args['batch_size'],
                                pch_size=args['patch_size'],
                                mask=True),
                     FakeTrain(path_list=path_list_fake,
                               length=num_iters_fake*args['batch_size'],
                               pch_size=args['patch_size'])]
    datasets = {'train':uData.ConcatDataset(datasets_list), 'val':BenchmarkTest(args['SIDD_test_h5'])}

    # train model
    print('\nBegin training with GPU: ' + str(args['gpu_id']))
    train_model(net, netG, datasets, optimizer, scheduler, args)

if __name__ == '__main__':
    main()

