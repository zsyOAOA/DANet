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
from networks import UNetD, UNetG, DiscriminatorLinear, sample_generator
from datasets.DenoisingDatasets import BenchmarkTrain, BenchmarkTest
from math import ceil
from utils import *
from loss import mean_match, get_gausskernel, gradient_penalty
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
import shutil
import warnings
from pathlib import Path
import commentjson as json

# filter warnings
warnings.simplefilter('ignore', Warning, lineno=0)

# default dtype
torch.set_default_dtype(torch.float32)

_C = 3
_modes = ['train', 'val']

def train_step_P(net, x, y, optimizerP, args):
    alpha = args['alpha']
    batch_size =x.shape[0]
    # zero the gradient
    net['P'].zero_grad()
    # raal data
    real_data = torch.cat([x,y], 1)
    real_loss = net['P'](real_data).mean()
    # generator fake data
    with torch.autograd.no_grad():
        fake_y = sample_generator(net['G'], x)
        fake_y_data = torch.cat([x, fake_y], 1)
    fake_y_loss = net['P'](fake_y_data.data).mean()
    grad_y_loss = gradient_penalty(real_data, fake_y_data, net['P'], args['lambda_gp'])
    loss_y = alpha * (fake_y_loss - real_loss)
    loss_yg = alpha * grad_y_loss
    # Denoiser fake data
    with torch.autograd.no_grad():
        fake_x = y - net['D'](y)
        fake_x_data = torch.cat([fake_x, y], 1)
    fake_x_loss = net['P'](fake_x_data.data).mean()
    grad_x_loss = gradient_penalty(real_data, fake_x_data, net['P'], args['lambda_gp'])
    loss_x = (1-alpha) * (fake_x_loss - real_loss)
    loss_xg = (1-alpha) * grad_x_loss
    loss = loss_x + loss_xg + loss_y + loss_yg
    # backward
    loss.backward()
    optimizerP.step()

    return loss, loss_x, loss_xg, loss_y, loss_yg

def train_step_G(net, x, y, optimizerG, args):
    alpha = args['alpha']
    batch_size = x.shape[0]
    # zero the gradient
    net['G'].zero_grad()
    fake_y = sample_generator(net['G'], x)
    loss_mean = args['tau_G'] * mean_match(x, y, fake_y, kernel.to(x.device), _C)
    fake_y_data = torch.cat([x, fake_y], 1)
    fake_y_loss = net['P'](fake_y_data).mean()
    loss_y = -alpha * fake_y_loss
    loss = loss_y + loss_mean
    # backward
    loss.backward()
    optimizerG.step()

    return loss, loss_y, loss_mean, fake_y.data

def train_step_D(net, x, y, optimizerD, args):
    alpha = args['alpha']
    batch_size = x.shape[0]
    # zero the gradient
    net['D'].zero_grad()
    fake_x = y -net['D'](y)
    mae_loss = F.l1_loss(fake_x, x, reduction='mean')
    fake_x_data = torch.cat([fake_x, y], 1)
    fake_x_loss = net['P'](fake_x_data).mean()
    loss_x = -(1-alpha) * fake_x_loss
    loss_e =  args['tau_D'] * mae_loss
    loss = loss_x + loss_e
    # backward
    loss.backward()
    optimizerD.step()

    return loss, loss_x, loss_e, mae_loss, fake_x.data

def train_epoch(net, datasets, optimizer, lr_scheduler, args):
    batch_size = {'train':args['batch_size'], 'val':4}
    data_loader = {phase:uData.DataLoader(datasets[phase], batch_size=batch_size[phase],
                   shuffle=True, num_workers=args['num_workers'], pin_memory=True) for phase in _modes}
    num_data = {phase:len(datasets[phase]) for phase in _modes}
    num_iter_epoch = {phase: ceil(num_data[phase] / batch_size[phase]) for phase in _modes}
    step = args['step'] if args['resume'] else 0
    step_img = args['step_img'] if args['resume'] else {x:0 for x in _modes}
    writer = SummaryWriter(str(Path(args['log_dir'])))
    for epoch in range(args['epoch_start'], args['epochs']):
        loss_epoch = {x:0 for x in ['PL', 'DL', 'GL']}
        subloss_epoch = {x:0 for x in ['Px', 'Pxg', 'Py', 'Pyg', 'Dx', 'DE', 'DAE', 'Gy', 'GMean',
                                                                                   'GErr', 'TGErr']}
        mae_epoch = {'train':0, 'val':0}
        tic = time.time()
        # train stage
        net['D'].train()
        net['G'].train()
        net['P'].train()
        lr_D = optimizer['D'].param_groups[0]['lr']
        lr_G = optimizer['G'].param_groups[0]['lr']
        lr_P = optimizer['P'].param_groups[0]['lr']
        if lr_D < 1e-6:
            sys.exit('Reach the minimal learning rate')
        phase = 'train'
        iter_GD = 0
        for ii, data in enumerate(data_loader[phase]):
            im_noisy, im_gt = [x.cuda() for x in data]
            # update the netP
            PL, Px, Pxg, Py, Pyg = train_step_P(net, im_gt, im_noisy, optimizer['P'], args)
            loss_epoch['PL'] += PL.item()
            subloss_epoch['Px'] += Px.item()
            subloss_epoch['Pxg'] += Pxg.item()
            subloss_epoch['Py'] += Py.item()
            subloss_epoch['Pyg'] += Pyg.item()
            # update the netD
            if (ii+1) % args['num_critic'] == 0:
                DL, Dx, DE, DAE, im_denoise = train_step_D(net, im_gt, im_noisy, optimizer['D'], args)
                loss_epoch['DL'] += DL.item()
                subloss_epoch['Dx'] += Dx.item()
                subloss_epoch['DE'] += DE.item()
                subloss_epoch['DAE'] += DAE.item()
                mae_epoch[phase] += DAE.item()
                # update the netG
                GL, Gy, GMean, im_generate = train_step_G(net, im_gt, im_noisy, optimizer['G'], args)
                loss_epoch['GL'] += GL.item()
                subloss_epoch['Gy'] += Gy.item()
                subloss_epoch['GMean'] += GMean.item()
                GErr = F.l1_loss(im_generate, im_gt, reduction='mean')
                subloss_epoch['GErr'] += GErr.item()
                TGErr = F.l1_loss(im_noisy, im_gt, reduction='mean')
                subloss_epoch['TGErr'] += TGErr.item()
                iter_GD += 1

                if (ii+1) % args['print_freq'] ==0:
                    template = '[Epoch:{:>2d}/{:<3d}] {:s}:{:0>5d}/{:0>5d}, PLx:{:>6.2f}/{:4.2f},'+\
                                         ' PLy:{:>6.2f}/{:4.2f}, DL:{:>6.2f}/{:.1e}, DAE:{:.2e}, '+\
                                                            'GL:{:>6.2f}/{:<5.2f}, GErr:{:.1e}/{:.1e}'
                    print(template.format(epoch+1, args['epochs'], phase, ii+1, num_iter_epoch[phase],
                                 Px.item(), Pxg.item(), Py.item(), Pyg.item(), Dx.item(), DE.item(),
                                    DAE.item(), Gy.item(), GMean.item(), GErr.item(), TGErr.item()))
                    writer.add_scalar('Train PNet Loss Iter', PL.item(), step)
                    writer.add_scalar('Train DNet Loss Iter', DL.item(), step)
                    writer.add_scalar('Train GNet Loss Iter', GL.item(), step)
                    step += 1
                    if (ii+1) % (10*args['print_freq'])==0:
                        x1 = vutils.make_grid(im_noisy, normalize=True, scale_each=True)
                        writer.add_image(phase+' Noisy Image', x1, step_img[phase])
                        x2 = vutils.make_grid(im_gt, normalize=True, scale_each=True)
                        writer.add_image(phase+' GroundTruth', x2, step_img[phase])
                        x3 = vutils.make_grid(im_denoise.clamp_(0.0,1.0), normalize=True,
                                                                                    scale_each=True)
                        writer.add_image(phase+' Denoised images', x3, step_img[phase])
                        x4 = vutils.make_grid(im_generate.clamp_(0.0, 1.0), normalize=True,
                                                                                    scale_each=True)
                        writer.add_image(phase+' Generated images', x4, step_img[phase])
                        step_img[phase] += 1

        loss_epoch['PL'] /= (ii+1)
        subloss_epoch['Px'] /= (ii+1)
        subloss_epoch['Pxg'] /= (ii+1)
        subloss_epoch['Py'] /= (ii+1)
        subloss_epoch['Pyg'] /= (ii+1)
        loss_epoch['DL'] /= (iter_GD+1)
        subloss_epoch['Dx'] /= (iter_GD+1)
        subloss_epoch['DAE'] /= (iter_GD+1)
        mae_epoch[phase] /= (iter_GD +1)
        loss_epoch['GL'] /= (iter_GD+1)
        subloss_epoch['Gy'] /= (iter_GD+1)
        subloss_epoch['GMean'] /= (iter_GD+1)
        subloss_epoch['GErr'] /= (iter_GD+1)
        subloss_epoch['TGErr'] /= (iter_GD+1)
        template = '{:s}: PL={:5.2f}, DL={:5.2f}, GL={:5.2f}, DAE:{:4.2e}, GMean:{:4.2e}, ' +\
                                 'GE:{:.2e}/{:.2e}, tauDG:{:.1e}/{:.1e}, lrDGP:{:.2e}/{:.2e}/{:.2e}'
        print(template.format(phase, loss_epoch['PL'], loss_epoch['DL'], loss_epoch['GL'],
                                subloss_epoch['DAE'], subloss_epoch['GMean'], subloss_epoch['GErr'],
                            subloss_epoch['TGErr'], args['tau_D'], args['tau_G'], lr_D, lr_G, lr_P))
        print('-'*150)

        # test stage
        net['D'].eval()
        psnr_per_epoch = ssim_per_epoch = 0
        phase = 'val'
        for ii, data in enumerate(data_loader[phase]):
            im_noisy, im_gt = [x.cuda() for x in data]
            with torch.set_grad_enabled(False):
                im_denoise = im_noisy - net['D'](im_noisy)

            mae_iter = F.l1_loss(im_denoise, im_gt)
            im_denoise.clamp_(0.0, 1.0)
            mae_epoch[phase] += mae_iter
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
        mae_epoch[phase] /= (ii+1)
        print('{:s}: mae={:.3e}, PSNR={:4.2f}, SSIM={:5.4f}'.format(phase, mae_epoch[phase],
                                                                    psnr_per_epoch, ssim_per_epoch))
        print('-'*150)

        # adjust the learning rate
        lr_scheduler['D'].step()
        lr_scheduler['G'].step()
        lr_scheduler['P'].step()
        # save model
        model_prefix = 'model_'
        save_path_model = str(Path(args['model_dir']) / (model_prefix+str(epoch+1)))
        torch.save({
            'epoch': epoch+1,
            'step': step+1,
            'step_img': {x:step_img[x]+1 for x in _modes},
            'model_state_dict': {x: net[x].state_dict() for x in ['D', 'P', 'G']},
            'optimizer_state_dict': {x: optimizer[x].state_dict() for x in ['D', 'P', 'G']},
            'lr_scheduler_state_dict': {x: lr_scheduler[x].state_dict() for x in ['D', 'P', 'G']}
            }, save_path_model)
        model_prefix = 'model_state_'
        save_path_model = str(Path(args['model_dir']) / (model_prefix+str(epoch+1)+'.pt'))
        torch.save({x:net[x].state_dict() for x in ['D', 'G']}, save_path_model)

        writer.add_scalars('MAE_epoch', mae_epoch, epoch)
        writer.add_scalar('Val PSNR epoch', psnr_per_epoch, epoch)
        writer.add_scalar('Val SSIM epoch', ssim_per_epoch, epoch)
        toc = time.time()
        print('This epoch take time {:.2f}'.format(toc-tic))
    writer.close()
    print('Reach the maximal epochs! Finish training')

def main():
    # set parameters
    with open('./configs/DANet.json', 'r') as f:
        args = json.load(f)

    # set the available GPUs
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args['gpu_id'])

    # build up the denoiser
    netD= UNetD(_C, wf=args['wf'], depth=args['depth']).cuda()
    # build up the generator
    netG= UNetG(_C, wf=args['wf'], depth=args['depth']).cuda()
    # build up the discriminator
    netP = DiscriminatorLinear(_C*2, ndf=args['ndf']).cuda()
    net = {'D':netD, 'G':netG, 'P':netP}

    # optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=args['lr_D'])
    optimizerG = optim.Adam(netG.parameters(), lr=args['lr_G'], betas=(0.5, 0.90))
    optimizerP = optim.Adam(netP.parameters(), lr=args['lr_P'], betas=(0.5, 0.90))
    optimizer = {'D':optimizerD, 'G':optimizerG, 'P':optimizerP}
    # schular
    schedulerD = optim.lr_scheduler.MultiStepLR(optimizerD, args['milestones'], gamma=0.5)
    schedulerG = optim.lr_scheduler.MultiStepLR(optimizerG, args['milestones'], gamma=0.5)
    schedulerP = optim.lr_scheduler.MultiStepLR(optimizerP, args['milestones'], gamma=0.5)
    scheduler = {'D':schedulerD, 'G':schedulerG, 'P':schedulerP}

    if args['resume']:
        if Path(args['resume']).is_file():
            print('=> Loading checkpoint {:s}'.format(str(Path(args['resume']))))
            checkpoint = torch.load(str(Path(args['resume'])), map_location='cpu')
            args['epoch_start'] = checkpoint['epoch']
            args['step'] = checkpoint['step']
            args['step_img'] = checkpoint['step_img']
            optimizerD.load_state_dict(checkpoint['optimizer_state_dict']['D'])
            optimizerG.load_state_dict(checkpoint['optimizer_state_dict']['G'])
            optimizerP.load_state_dict(checkpoint['optimizer_state_dict']['P'])
            schedulerD.load_state_dict(checkpoint['lr_scheduler_state_dict']['D'])
            schedulerG.load_state_dict(checkpoint['lr_scheduler_state_dict']['G'])
            schedulerP.load_state_dict(checkpoint['lr_scheduler_state_dict']['P'])
            netD.load_state_dict(checkpoint['model_state_dict']['D'])
            netG.load_state_dict(checkpoint['model_state_dict']['G'])
            netP.load_state_dict(checkpoint['model_state_dict']['P'])
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
    datasets = {'train':BenchmarkTrain(h5_file=args['SIDD_train_h5'],
                                       length=5000*args['batch_size']*args['num_critic'],
                                       pch_size=args['patch_size'],
                                       mask=False),
                'val':BenchmarkTest(args['SIDD_test_h5'])}

    # build the Gaussian kernel for loss
    global kernel
    kernel = get_gausskernel(args['ksize'], chn=_C)

    # train model
    print('\nBegin training with GPU: ' + str(args['gpu_id']))
    train_epoch(net, datasets, optimizer, scheduler, args)

if __name__ == '__main__':
    main()

