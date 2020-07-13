#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-04-08 16:24:58

import os
import numpy as np
from cv2 import imread, imwrite
from glob import glob
import random
from skimage import img_as_ubyte, img_as_float
import h5py as h5
import argparse

parser = argparse.ArgumentParser(prog='Renoir dataset Generation')
# The aligned Renoir images: /ssd1t/Renoir/
parser.add_argument('--data_dir', default=None, type=str, metavar='PATH',
                                   help="path to save the aligned Renoir Datasets, (default: None)")
args = parser.parse_args()

camera_type = ['Mi3_Aligned', 'S90_Aligned', 'T3i_Aligned']
print('Begin to make Groundtruth')
pair_list = []
for camera in camera_type:
    print('Processing the camera:{:s}'.format(camera))
    camera_dir = os.path.join(args.data_dir,  camera)
    batch_dir = [os.path.join(camera_dir, x) for x in os.listdir(camera_dir)]
    batch_dir = sorted(batch_dir)
    num_im = 0
    for im_dir in batch_dir:
        if os.path.isdir(im_dir):
            ref_path = glob(os.path.join(im_dir, '*Reference.bmp'))
            ref_path.extend(glob(os.path.join(im_dir, '*full.bmp')))
            im_gt = sum([img_as_float(imread(x, 1)) for x in ref_path]) / len(ref_path)
            im_gt = img_as_ubyte(im_gt)
            gt_path = os.path.join(im_dir, 'groundTruth.bmp')
            if os.path.exists(gt_path):
                os.remove(gt_path)
            imwrite(gt_path, im_gt)
            noisy_path_ls = glob(os.path.join(im_dir, '*Noisy.bmp'))
            for noisy_path in noisy_path_ls:
                pair_list.append({'noisy_path':noisy_path,'gt_path':gt_path})
                num_im +=1
    print('The camera {:s} has {:d} image pairs\n'.format(camera[:-1], num_im))

pch_size = 512
stride = 512-128
print('\nSaving the images to Hdf5 Format')
small_img_h5 = os.path.join(args.data_dir, 'small_imgs_all.hdf5')
num_patch = 0
if os.path.exists(small_img_h5):
    os.remove(small_img_h5)
with h5.File(small_img_h5, 'w') as h5_file:
    for ii, path in enumerate(pair_list):
        if (ii+1) % 20 == 0:
            print('{:d}/{:d}'.format(ii+1, len(pair_list)))
        im_noisy = imread(path['noisy_path'])[:, :, ::-1]
        im_gt = imread(path['gt_path'])[:, :, ::-1]
        H, W, _ = im_gt.shape
        ind_H = list(range(0, H-pch_size+1, stride))
        if ind_H[-1] < H-pch_size:
            ind_H.append(H-pch_size)
        ind_W = list(range(0, W-pch_size+1, stride))
        if ind_W[-1] < W-pch_size:
            ind_W.append(W-pch_size)
        for start_H in ind_H:
            for start_W in ind_W:
                pch_noisy = im_noisy[start_H:start_H+pch_size, start_W:start_W+pch_size, ]
                pch_gt = im_gt[start_H:start_H+pch_size, start_W:start_W+pch_size, ]
                pchs = np.concatenate((pch_noisy, pch_gt), 2)
                h5_file.create_dataset(name=str(num_patch), shape=pchs.shape, dtype=pchs.dtype,
                                                                                     data=pchs)
                num_patch += 1
print('Number of patches: {:d}'.format(num_patch))

