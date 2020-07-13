#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-09-02 15:35:24

import random
import numpy as np
import torch.utils.data as uData
import h5py as h5
import cv2

# Base Datasets
class BaseDataSetH5(uData.Dataset):
    def __init__(self, h5_path, length=None):
        '''
        Args:
            h5_path (str): path of the hdf5 file
            length (int): length of Datasets
        '''
        super(BaseDataSetH5, self).__init__()
        self.h5_path = h5_path
        self.length = length
        with h5.File(h5_path, 'r') as h5_file:
            self.keys = list(h5_file.keys())
            self.num_images = len(self.keys)

    def __len__(self):
        if self.length == None:
            return self.num_images
        else:
            return self.length

    def crop_patch(self, imgs_sets):
        H, W, C2 = imgs_sets.shape
        # minus the bayer patter channel
        C = int(C2/2)
        ind_H = random.randint(0, H-self.pch_size)
        ind_W = random.randint(0, W-self.pch_size)
        im_noisy = np.array(imgs_sets[ind_H:ind_H+self.pch_size, ind_W:ind_W+self.pch_size, :C])
        im_gt = np.array(imgs_sets[ind_H:ind_H+self.pch_size, ind_W:ind_W+self.pch_size, C:])
        return im_gt, im_noisy

class BaseDataSetFolder(uData.Dataset):
    def __init__(self, path_list, pch_size, length=None):
        '''
        Args:
            path_list (str): path of the hdf5 file
            length (int): length of Datasets
        '''
        super(BaseDataSetFolder, self).__init__()
        self.path_list = path_list
        self.length = length
        self.pch_size = pch_size
        self.num_images = len(path_list)

    def __len__(self):
        if self.length == None:
            return self.num_images
        else:
            return self.length

    def crop_patch(self, im):
        pch_size = self.pch_size
        H, W, _ = im.shape
        if H < self.pch_size or W < self.pch_size:
            H = max(pch_size, H)
            W = max(pch_size, W)
            im = cv2.resize(im, (W, H))
        ind_H = random.randint(0, H-pch_size)
        ind_W = random.randint(0, W-pch_size)
        im_pch = im[ind_H:ind_H+pch_size, ind_W:ind_W+pch_size,]
        return im_pch
