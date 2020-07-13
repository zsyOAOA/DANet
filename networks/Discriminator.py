#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-09-18 22:31:45

import torch.nn as nn
import torch.nn.init as init
import torch.nn.utils as utils
import torch.nn.functional as F
from .SubBlocks import conv_down

class DiscriminatorLinear(nn.Module):
    def __init__(self, in_chn, ndf=64, slope=0.2):
        '''
        ndf: number of filters
        '''
        super(DiscriminatorLinear, self).__init__()
        self.ndf = ndf
        # input is N x C x 128 x 128
        main_module = [conv_down(in_chn, ndf, bias=False),
                       nn.LeakyReLU(slope, inplace=True)]
        # state size: N x ndf x 64 x 64
        main_module.append(conv_down(ndf, ndf*2, bias=False))
        main_module.append(nn.LeakyReLU(slope, inplace=True))
        # state size: N x (ndf*2) x 32 x 32
        main_module.append(conv_down(ndf*2, ndf*4, bias=False))
        main_module.append(nn.LeakyReLU(slope, inplace=True))
        # state size: N x (ndf*4) x 16 x 16
        main_module.append(conv_down(ndf*4, ndf*8, bias=False))
        main_module.append(nn.LeakyReLU(slope, inplace=True))
        # state size: N x (ndf*8) x 8 x 8
        main_module.append(conv_down(ndf*8, ndf*16, bias=False))
        main_module.append(nn.LeakyReLU(slope, inplace=True))
        # state size: N x (ndf*16) x 4 x 4
        main_module.append(nn.Conv2d(ndf*16, ndf*32, 4, stride=1, padding=0, bias=False))
        main_module.append(nn.LeakyReLU(slope, inplace=True))
        # state size: N x (ndf*32) x 1 x 1
        self.main = nn.Sequential(*main_module)
        self.output = nn.Linear(ndf*32, 1)

        self._initialize()

    def forward(self, x):
        feature = self.main(x)
        feature = feature.view(-1, self.ndf*32)
        out = self.output(feature)
        return out.view(-1)

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.normal_(m.weight.data, 0., 0.02)
                if not m.bias is None:
                    init.constant_(m.bias, 0)

