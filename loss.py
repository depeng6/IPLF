# -*- encoding: utf-8 -*-
"""
@File    : loss.py
@Time    : 2021/9/23 16:10
@Author  : Chool
@Email   : depeng_cust@163.com
@Software: PyCharm
"""
import math

import torch
import torch.nn.functional as F
import torch.nn as nn
from math import exp
from torch.autograd import Variable
import numpy as np
import cv2 as cv
from torchvision import  transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()



def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)


    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2


    delta1 = torch.sqrt(sigma1_sq*sigma2_sq+0.000001)


    C1 = 0.01**2
    C2 = 0.03**2

    lum = (2*mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    con = (2*delta1 + C2) / (sigma1_sq + sigma2_sq + C2)
    str = (sigma12 + C2) / (delta1 + C2)


    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean(),lum.mean(),con.mean(),str.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def image_edge(image):
    image = image
    b,c,h, w = image.shape
    k = torch.tensor([-1, 0, 0, 1], dtype=torch.float).cuda()
    k = k.view(1, 1, 2, 2)

    z = F.conv2d(image, k, padding=0)
    return z


