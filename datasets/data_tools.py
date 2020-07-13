#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-09-02 15:11:05

import cv2
import numpy as np
import random
from math import ceil

def data_augmentation(image, mode):
    '''
    Performs data augmentation of the input image
    Input:
        image: a cv2 (OpenCV) image
        mode: int. Choice of transformation to apply to the image
                0 - no transformation
                1 - flip up and down
                2 - rotate counterwise 90 degree
                3 - rotate 90 degree and flip up and down
                4 - rotate 180 degree
                5 - rotate 180 degree and flip
                6 - rotate 270 degree
                7 - rotate 270 degree and flip
    '''
    if mode == 0:
        # original
        out = image
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')

    return out

def inverse_data_augmentation(image, mode):
    '''
    Performs inverse data augmentation of the input image
    '''
    if mode == 0:
        # original
        out = image
    elif mode == 1:
        out = np.flipud(image)
    elif mode == 2:
        out = np.rot90(image, axes=(1,0))
    elif mode == 3:
        out = np.flipud(image)
        out = np.rot90(out, axes=(1,0))
    elif mode == 4:
        out = np.rot90(image, k=2, axes=(1,0))
    elif mode == 5:
        out = np.flipud(image)
        out = np.rot90(out, k=2, axes=(1,0))
    elif mode == 6:
        out = np.rot90(image, k=3, axes=(1,0))
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.flipud(image)
        out = np.rot90(out, k=3, axes=(1,0))
    else:
        raise Exception('Invalid choice of image transformation')

    return out

def random_augmentation(*args):
    out = []
    if random.randint(0,1) == 1:
        flag_aug = random.randint(1,7)
        for data in args:
            out.append(data_augmentation(data, flag_aug).copy())
    else:
        for data in args:
            out.append(data)
    return out

def im_pad_fun(image, offset):
    '''
    Input:
        image: numpy array, H x W x C
    '''
    H, W, C = image.shape
    if (H % offset == 0) and (W % offset == 0):
        image_pad = image
    else:
        H_pad = H if (H % offset == 0) else (offset * ceil(H / offset))
        W_pad = W if (W % offset == 0) else (offset * ceil(W / offset))
        image_pad = np.zeros([H_pad, W_pad, C], dtype=image.dtype)
        image_pad[:H, :W] = image

        if (H % offset) != 0:
            image_pad[H:, :W] = image[(H%offset-offset):, ][::-1,]

        if (W % offset) != 0:
            image_pad[:, W:] = image_pad[:, (W-(offset-W%offset)):W][:, ::-1]

    return image_pad

if __name__ == '__main__':
    # aa = np.random.randn(4,4)
    # for ii in range(8):
        # bb1 = data_augmentation(aa, ii)
        # bb2 = inverse_data_augmentation(bb1, ii)
        # if np.allclose(aa, bb2):
            # print('Flag: {:d}, Sccessed!'.format(ii))
        # else:
            # print('Flag: {:d}, Failed!'.format(ii))

    aa = np.random.randn(6, 6, 3)
    bb = im_pad_fun(aa, 3)
    print(aa[:, :, 0])
    print(bb[:, :, 0])


