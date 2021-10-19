# -*- encoding: utf-8 -*-
"""
@File    : net.py
@Time    : 2021/9/22 21:07
@Author  : Chool
@Email   : depeng_cust@163.com
@Software: PyCharm
"""
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter



def conv3x3(in_channels,out_channels,stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,
                     padding=1,bias=False)

def conv1x1(in_channels,out_channels,stride=1):
    "1x1 convolution"
    return nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,
                     bias=False)


class Spatial_Attention(nn.Module):
    def __init__(self,in_channels, kernel_size = 7, ratio = 8):
        super(Spatial_Attention, self).__init__()
        self.conv1_1 = conv1x1(in_channels,in_channels//ratio)
        self.conv1_2 = conv1x1(in_channels,in_channels//ratio)
        self.act_f = nn.Hardswish()
        self.conv2 = nn.Conv2d(in_channels//ratio, 4, kernel_size, padding=3, bias=False)

    def forward(self,x):
        out1_1 = self.conv1_1(x)
        out1_2 = self.conv1_2(x)
        out1_1 = self.act_f(out1_1)
        out1_2 = self.act_f(out1_2)
        out2 = out1_1 * out1_2
        out2 = self.conv2(out2)
        out2, _ = torch.max(out2,dim=1,keepdim=True)
        return out2


class BaseBlock_1(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1):
        super(BaseBlock_1, self).__init__()

        self.conv1 = conv3x3(in_channels,out_channels,stride)
        self.conv2 = conv3x3(out_channels,out_channels,stride)
        self.act_f = nn.Sigmoid()
        # self.act_f = nn.ReLU()


        if in_channels != out_channels:
            self.downsample = conv1x1(in_channels,out_channels)
        else:
            self.downsample = lambda x: x


    def forward(self,x):
        res = self.downsample(x)
        out = self.act_f(x)

        out = self.conv1(out)


        out = self.act_f(out)

        out = self.conv2(out)

        out += res

        return out

class Fusion_Net(nn.Module):
    def __init__(self, out_channels):
        super(Fusion_Net, self).__init__()
        filter_n = [16,32,64]
        # self.conv1 = BaseBlock_3(2,filter_n[0])
        self.conv1 = nn.Conv2d(2,filter_n[0],kernel_size=7,stride=1,padding=3)

        self.conv2 = BaseBlock_1(filter_n[0],filter_n[2])

        self.conv3 = BaseBlock_1(filter_n[2],filter_n[2])

        self.conv4 = BaseBlock_1(filter_n[2],filter_n[2])
        self.sa = Spatial_Attention(filter_n[2]*3)

        self.conv5 = BaseBlock_1(filter_n[2]*3,out_channels)

    def forward(self,ir,vis):
        out_cat = torch.cat((ir,vis),dim = 1)
        out1 = self.conv1(out_cat)

        out2 = self.conv2(out1)

        out3 = self.conv3(out2)

        out4 = self.conv4(out3)

        out4 = torch.cat((out2,out3,out4),dim=1)
        out4 = self.sa(out4) * out4

        out5 = self.conv5(out4)

        return out5



if __name__ == '__main__':

    image1 = torch.rand(32,1,80,80)
    image2 = torch.rand(32,1,80,80)

    fuse = Fusion_Net(1)
    out1 = fuse(image1,image2)

    print(out1.shape)







