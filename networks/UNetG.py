#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-03-20 19:48:14
# Adapted from https://github.com/jvanvugt/pytorch-unet

import torch
from torch import nn
import torch.nn.functional as F
from .SubBlocks import conv3x3, conv_down
from .UNetD import UNetD

class UNetG(UNetD):
    def __init__(self, in_chn, wf=32, depth=5, relu_slope=0.20):
        """
        Reference:
        Ronneberger O., Fischer P., Brox T. (2015) U-Net: Convolutional Networks for Biomedical
        Image Segmentation. MICCAI 2015.
        ArXiv Version: https://arxiv.org/abs/1505.04597

        Args:
            in_chn (int): number of input channels, Default 3
            depth (int): depth of the network, Default 4
            wf (int): number of filters in the first layer, Default 32
        """
        super(UNetG, self).__init__(in_chn, wf, depth, relu_slope)

    def get_input_chn(self, in_chn):
        return in_chn+1

def sample_generator(netG, x):
    z = torch.randn([x.shape[0], 1, x.shape[2], x.shape[3]], device=x.device)
    x1 = torch.cat([x, z], dim=1)
    out = netG(x1)

    return out+x
