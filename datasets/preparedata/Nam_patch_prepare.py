#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2020-02-17 13:10:35

import random
import argparse
import numpy as np
from pathlib import Path
from math import ceil
from scipy.io import loadmat, savemat

random.seed(0)

class PatchCrop:
    def __init__(self, pch_size, H, W):
        assert (pch_size < H and pch_size < W)
        self.ind_H = random.randint(0, H-pch_size)
        self.ind_W = random.randint(0, W-pch_size)

    def crop(self, img):
        ind_H = self.ind_H
        ind_W = self.ind_W
        return img[ind_H:ind_H+pch_size, ind_W:ind_W+pch_size,]

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/ssd1t/Nam/', type=str, metavar='PATH',
                                          help="Path to save the Nam dataset, (default: None)")
args = parser.parse_args()

num_pch = 100
save_path = Path(args.data_dir) / 'patch{:04d}.mat'.format(num_pch)
if save_path.exists():
    save_path.unlink()

pch_size = 512
mat_list = sorted([str(x) for x in Path(args.data_dir).glob('**/*.mat')])
num_img = len(mat_list)
pch_per_img = ceil(num_pch / num_img)

noisy_pchs = np.zeros([num_pch, 512, 512, 3], dtype=np.uint8)
gt_pchs = np.zeros([num_pch, 512, 512, 3], dtype=np.uint8)

iter_pch = 0
for ii, mat_path in enumerate(mat_list):
    print('Image: {:02d}, path: {:s}'.format(ii+1, mat_path))
    data_dict = loadmat(mat_path)
    noisy_img = data_dict['img_noisy']
    gt_img = data_dict['img_mean']
    H, W, _ = gt_img.shape
    for jj in range(pch_per_img):
        patchcrop = PatchCrop(pch_size, H, W)
        noisy_pchs[iter_pch, ] = patchcrop.crop(noisy_img)
        gt_pchs[iter_pch, ] = patchcrop.crop(gt_img)
        iter_pch += 1
        if iter_pch ==  num_pch:
            break
    del data_dict

savemat(str(save_path), {'noisy_pchs':noisy_pchs, 'gt_pchs':gt_pchs})

