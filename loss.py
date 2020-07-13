#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-10-31 21:31:50

import torch
import torch.nn.functional as F
import cv2
import numpy as np

def gradient_penalty(real_data, generated_data, netP, lambda_gp):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        alpha = alpha.to(real_data.device)
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated.requires_grad=True

        # Calculate probability of interpolated examples
        prob_interpolated = netP(interpolated)

        # Calculate gradients of probabilities with respect to examples
        grad_outputs = torch.ones(prob_interpolated.size(), device=real_data.device, dtype=torch.float32)
        gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                 grad_outputs=grad_outputs, create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return  lambda_gp * ((gradients_norm - 1) ** 2).mean()

def get_gausskernel(p, chn=3):
    '''
    Build a 2-dimensional Gaussian filter with size p
    '''
    x = cv2.getGaussianKernel(p, sigma=-1)   # p x 1
    y = np.matmul(x, x.T)[np.newaxis, np.newaxis,]  # 1x 1 x p x p
    out = np.tile(y, (chn, 1, 1, 1)) # chn x 1 x p x p

    return torch.from_numpy(out).type(torch.float32)

def gaussblur(x, kernel, p=5, chn=3):
    x_pad = F.pad(x, pad=[int((p-1)/2),]*4, mode='reflect')
    y = F.conv2d(x_pad, kernel, padding=0, stride=1, groups=chn)

    return y

def var_match(x, y, fake_y, kernel, chn=3):
    p = kernel.shape[2]
    # estimate the real distribution
    err_real = y - x
    mu_real = gaussblur(err_real, kernel, p, chn)
    err2_real = (err_real-mu_real)**2
    var_real = gaussblur(err2_real, kernel, p, chn)
    var_real = torch.where(var_real<1e-10, torch.ones_like(fake_y)*1e-10, var_real)
    # estimate the fake distribution
    err_fake = fake_y - x
    mu_fake = gaussblur(err_fake, kernel, p, chn)
    err2_fake = (err_fake-mu_fake)**2
    var_fake = gaussblur(err2_fake, kernel, p, chn)
    var_fake = torch.where(var_fake<1e-10, torch.ones_like(fake_y)*1e-10, var_fake)

    # calculate the loss
    loss_err = F.l1_loss(mu_real, mu_fake, reduction='mean')
    loss_var = F.l1_loss(var_real, var_fake, reduction='mean')

    return loss_err, loss_var

def mean_match(x, y, fake_y, kernel, chn=3):
    p = kernel.shape[2]
    # estimate the real distribution
    err_real = y - x
    mu_real = gaussblur(err_real, kernel, p, chn)
    err_fake = fake_y - x
    mu_fake = gaussblur(err_fake, kernel, p, chn)
    loss = F.l1_loss(mu_real, mu_fake, reduction='mean')

    return loss

