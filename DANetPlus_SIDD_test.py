#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-09-20 19:27:43

import torch
import argparse
import numpy as np
from os.path import join
from networks import UNetD
from scipy.io import loadmat, savemat
from skimage import img_as_float32, img_as_ubyte
from datasets.data_tools import data_augmentation, inverse_data_augmentation

def str2bool(v):
    return v.lower() in ('true')

parser = argparse.ArgumentParser(prog='VDN Test', description='optional parameters for test')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                                                help="path to the saved checkpoint (default: None)")
parser.add_argument('--flip', default='False', type=str2bool,
                                                  help="Data ensemble when testing (default: None)")
parser.add_argument('--save_path', default='', type=str, metavar='PATH',
                                             help="path to save the denoise result (default: None)")
args = parser.parse_args()

noisy_mat = loadmat('/ssd1t/SIDD/BenchmarkNoisyBlocksSrgb.mat')['BenchmarkNoisyBlocksSrgb']
num_images, num_blocks, H, W, C = noisy_mat.shape

denoise_res = np.zeros_like(noisy_mat)

# load the model
net = UNetD(3)
net = torch.nn.DataParallel(net).cuda()
net.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))

net.eval()
total_time = 0
for ii in range(num_images):
    print('The {:d} image'.format(ii+1))
    for jj in range(num_blocks):
        pch_noisy = img_as_float32(noisy_mat[ii,jj,])  # 256 x 256 x 3
        if args.flip:
            pch_denoise = np.zeros_like(pch_noisy)
            for flag in range(8):
                pch_noisy_flag = np.ascontiguousarray(data_augmentation(pch_noisy, flag))
                pch_noisy_flag = torch.from_numpy(pch_noisy_flag.transpose((2,0,1))[np.newaxis,]).cuda()
                with torch.no_grad():
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    pch_denoise_flag = pch_noisy_flag - net(pch_noisy_flag)
                    end.record()
                    torch.cuda.synchronize()
                    total_time += start.elapsed_time(end)/1000
                pch_denoise_flag = pch_denoise_flag.cpu().numpy().squeeze().transpose((1,2,0))
                pch_denoise += inverse_data_augmentation(pch_denoise_flag, flag)
            pch_denoise /= 8
        else:
            pch_noisy_temp = torch.from_numpy(pch_noisy.transpose((2,0,1))[np.newaxis,]).cuda()
            with torch.no_grad():
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                pch_denoise = pch_noisy_temp - net(pch_noisy_temp)
                end.record()
                torch.cuda.synchronize()
                total_time += start.elapsed_time(end)/1000
            pch_denoise = pch_denoise.cpu().numpy().squeeze().transpose((1,2,0))
        denoise_res[ii, jj] = img_as_ubyte(pch_denoise.clip(0.0, 1.0))

megatime = total_time * 1024 * 1024 / (num_images*num_blocks*256*256)
if args.flip:
    save_path = join(args.save_path, 'BiANet+_SIDD_test_flip.mat')
else:
    save_path = join(args.save_path, 'BiANet+_SIDD_test_noflip.mat')
savemat(save_path, {'denoise_res':denoise_res, 'time':megatime})

