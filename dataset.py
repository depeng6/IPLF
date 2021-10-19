# -*- encoding: utf-8 -*-
"""
@File    : dataset.py
@Time    : 2021/9/23 15:10
@Author  : Chool
@Email   : depeng_cust@163.com
@Software: PyCharm
"""
import os
from os import listdir
from os.path import join
import numpy as np
import torch
from PIL import Image
from config import conf
from scipy.misc import imread, imsave, imresize
import cv2 as cv
from torchvision import  transforms
from torch.utils.data import Dataset,DataLoader
from torchvision import utils as vutils
from random import random
import matplotlib.pyplot as plt
from loss import ssim
import torch.nn.functional as F



def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


class Fusion_Dataset(Dataset):
    def __init__(self,image_path_ir,random_data=True):
        super(Fusion_Dataset,self).__init__()
        self.image_path_ir = image_path_ir
        self.image_name_ir = sorted(os.listdir(image_path_ir))
        self.image_name_ir  = self.image_name_ir[:1000]

        self.random_data = random_data

    def rand(self,a=0, b=1):
        return np.random.rand() * (b - a) + a

    def __len__(self):
        return len(self.image_name_ir)

    def __getitem__(self, index):
        image_name_ir = self.image_name_ir[index]
        trans = transforms.Compose([

            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485],
                std=[0.229])
        ])
        ps = torch.nn.PixelShuffle(16)
        Bright = transforms.Compose([transforms.ColorJitter(brightness=(0.7, 1.3), hue=0), ])
        GStrans5 = transforms.Compose([transforms.GaussianBlur(kernel_size=5, sigma=(1, 1000)), ])
        GStrans7 = transforms.Compose([transforms.GaussianBlur(kernel_size=7, sigma=(1, 1000)), ])
        GStrans11 = transforms.Compose([transforms.GaussianBlur(kernel_size=11, sigma=(1, 1000)), ])
        GStrans15 = transforms.Compose([transforms.GaussianBlur(kernel_size=15, sigma=(1, 1000)), ])
        sub = [GStrans5, GStrans7, GStrans11, GStrans15]
        GS = transforms.Compose([transforms.RandomChoice(sub)])
        black = transforms.Compose([transforms.RandomErasing(scale=(0.1, 0.2))])
        noise = torch.randn(1, 128, 128) * 0.1
        image_gt = Image.open(os.path.join(self.image_path_ir, image_name_ir)).convert('L')


        image_gt = trans(image_gt)

        C1 = image_gt.clone()
        C2 = image_gt.clone()

        a = 0.4
        b = 1
        c = 1
        if random() > a:
            mask = torch.rand(1,1,8, 8)
            mask = torch.repeat_interleave(mask, 256, dim=1)
            mask = ps(mask)
            mask = torch.where(mask > 0.5, 1, 0)
            C1 = GS(C1) * mask + C1 * (1 - mask)
            C1 = C1.squeeze(0)
        if random() > b:
            mask = torch.rand(1,1,8, 8)
            mask = torch.repeat_interleave(mask, 256, dim=1)
            mask = ps(mask)
            mask = torch.where(mask > 0.5, 1, 0)
            C1 = Bright(C1) * mask + C1 * (1 - mask)
            C1 = C1.squeeze(0)
        if random() > c:
            mask = torch.rand(1,1,8, 8)
            mask = torch.repeat_interleave(mask, 256, dim=1)
            mask = ps(mask)
            mask = torch.where(mask > 0.5, 1, 0)
            C1 = C1 + noise * (1 - mask)
            C1 = C1.squeeze(0)

        if random() > a:
            mask = torch.rand(1,1,8, 8)
            mask = torch.repeat_interleave(mask, 256, dim=1)
            mask = ps(mask)
            mask = torch.where(mask > 0.5, 1, 0)
            C2 = GS(C2) * mask + C2 * (1 - mask)
            C2 = C2.squeeze(0)
        if random() > b:
            mask = torch.rand(1,1,8, 8)
            mask = torch.repeat_interleave(mask, 256, dim=1)
            mask = ps(mask)
            mask = torch.where(mask > 0.5, 1, 0)
            C2 = Bright(C2) * mask + C2 * (1 - mask)
            C2 = C2.squeeze(0)
        if random() > c:
            mask = torch.rand(1,1,8, 8)
            mask = torch.repeat_interleave(mask, 256, dim=1)
            mask = ps(mask)
            mask = torch.where(mask > 0.5, 1, 0)
            C2 = C2 + noise * (1 - mask)
            C2 = C2.squeeze(0)
        return C1,C2,image_gt




def save_valid_image(image,epoch,batch):
    save_path = "valid_image/{}/Epoch_{}".format(conf.model_name,epoch)
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    vutils.save_image(image,os.path.join(save_path, 'fusion_'f'{batch+1}.png'))

def list_images(directory):
    images = []
    names = []
    dir = listdir(directory)
    dir.sort(key=lambda x:int(x[3:-4]))
    for file in dir:
        name = file.lower()
        if name.endswith('.png'):
            images.append(join(directory, file))
        if name.endswith('.bmp'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
        elif name.endswith('.tif'):
            images.append(join(directory, file))
        name1 = name.split('.')
        names.append(name1[0])
    return images

# load training images
def load_dataset(image_path, BATCH_SIZE, num_imgs=None):
    if num_imgs is None:
        num_imgs = len(image_path)
    original_imgs_path = image_path[:num_imgs]

    batches = int(len(original_imgs_path) // BATCH_SIZE)
    return original_imgs_path, batches


def get_train_images(paths):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = imread(path)
        if image.ndim == 3:
            image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        image = np.expand_dims(image,0)
        # print(image.ndim)
        image = image.astype(np.float32) / 255.0
        images.append(image)

    # print(images[0].size)
    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images













