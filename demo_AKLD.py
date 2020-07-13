#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2020-07-10 14:38:39

'''
In this demo, we only test the model on one image of SIDD validation dataset.
The full validation dataset can be download from the following website:
    https://www.eecs.yorku.ca/~kamel/sidd/benchmark.php
'''

import torch
import numpy as np
from networks import UNetG, sample_generator
from scipy.io import loadmat
from skimage import img_as_float32, img_as_ubyte
from matplotlib import pyplot as plt
from utils import PadUNet, kl_gauss_zero_center, estimate_sigma_gauss


# build the network
net = UNetG(3, wf=32, depth=5).cuda()

# load the pretrained model
net.load_state_dict(torch.load('./model_states/DANet.pt', map_location='cpu')['G'])

# read the images
im_noisy_real = loadmat('./test_data/SIDD/noisy.mat')['im_noisy']
im_gt = loadmat('./test_data/SIDD/gt.mat')['im_gt']


L = 50
AKLD = 0
im_noisy_real = torch.from_numpy(img_as_float32(im_noisy_real).transpose([2,0,1])).unsqueeze(0).cuda()
im_gt = torch.from_numpy(img_as_float32(im_gt).transpose([2,0,1])).unsqueeze(0).cuda()
sigma_real = estimate_sigma_gauss(im_noisy_real, im_gt)
with torch.autograd.no_grad():
    padunet = PadUNet(im_gt, dep_U=5)
    im_gt_pad = padunet.pad()
    for _ in range(L):
        outputs_pad = sample_generator(net, im_gt_pad)
        im_noisy_fake = padunet.pad_inverse(outputs_pad)
        im_noisy_fake.clamp_(0.0, 1.0)
        sigma_fake = estimate_sigma_gauss(im_noisy_fake, im_gt)
        kl_dis = kl_gauss_zero_center(sigma_fake, sigma_real)
        AKLD += kl_dis

AKLD /= L
print("AKLD value: {:.3f}".format(AKLD))


