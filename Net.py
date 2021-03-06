#====================================================
#  File Name   : Net.py
#  Author      : deffand
#  Date        : 2020/12/17
#  Description :
#====================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from Model import MyLoss,MonodepthLoss
from torch.autograd import Variable
import numpy as np


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride):
        super(Conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv_base = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=kernel_size, stride=stride)
        self.normalize = nn.BatchNorm2d(num_out_layers)

    def forward(self, x):
        p = int(np.floor((self.kernel_size-1)/2))
        p2d = (p, p, p, p)
        x = self.conv_base(F.pad(x, p2d))
        x = self.normalize(x)
        return F.elu(x, inplace=True)

class Upconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
        super(Upconv, self).__init__()
        self.scale = scale
        self.conv1 = Conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        return self.conv1(x)


class get_disp(nn.Module):
    def __init__(self, num_in_layers):
        super(get_disp, self).__init__()
        self.conv1 = nn.Conv2d(num_in_layers, 2, kernel_size=3, stride=1)
        self.normalize = nn.BatchNorm2d(2)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        p = 1
        p2d = (p, p, p, p)
        x = self.conv1(F.pad(x, p2d))
        x = self.normalize(x)
        return 0.3 * self.sigmoid(x)

class MyNet(nn.Module):

    def __init__(self, args, block):
        super(MyNet, self).__init__()
        self.mode = args.mode
        self.MyLoss = MyLoss(args)
        self.inplanes = 64
        self.conv_pre = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, 3, stride=2)
        self.layer2 = self._make_layer(block, 128, 4, stride=2)
        self.layer3 = self._make_layer(block, 256, 6, stride=2)
        self.layer4 = self._make_layer(block, 512, 3, stride=2)

        self.upconv6 = Upconv(512, 512, 3, 2)
        self.iconv6 = Conv(512 + 256, 512, 3, 1)

        self.upconv5 = Upconv(512, 256, 3, 2)
        self.iconv5 = Conv(256 + 128, 256, 3, 1)

        self.upconv4 = Upconv(256, 128, 3, 2)
        self.iconv4 = Conv(128 + 64, 128, 3, 1)
        self.disp4_layer = get_disp(128)

        self.upconv3 = Upconv(128, 64, 3, 2)
        self.iconv3 = Conv(64 + 64 + 2, 64, 3, 1)
        self.disp3_layer = get_disp(64)

        self.upconv2 = Upconv(64, 32, 3, 2)
        self.iconv2 = Conv(32 + 64 + 2, 32, 3, 1)
        self.disp2_layer = get_disp(32)

        self.upconv1 = Upconv(32, 16, 3, 2)
        self.iconv1 = Conv(16 + 2, 16, 3, 1)
        self.disp1_layer = get_disp(16)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def conv(self, inplanes, planes, kernel_size, stride):
        conv = nn.Sequential(nn.Conv2d(inplanes, planes,
                        kernel_size=kernel_size, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
        )
        self.inplanes = planes
        return conv

    def upconv(self, planes, kernel_size, stride,padding=0):
        conv = nn.Sequential(nn.Conv2d(self.inplanes, planes,
                        kernel_size=kernel_size, stride=stride, bias=False,padding=padding),
                nn.BatchNorm2d(planes),
        )
        self.inplanes = planes
        return conv

    def upsample(self, x, ratio):
        s = x.size()
        h = int(s[2])
        w = int(s[3])
        nh = h * ratio
        nw = w * ratio
        temp = nn.functional.upsample(x, [nh, nw], mode='nearest')
        return temp

    def disp_block(self, inplanes, planes=2):
        conv = nn.Sequential(nn.Conv2d(inplanes, planes,
                        kernel_size=3, stride=1, bias=False,padding=2),
                nn.BatchNorm2d(planes),
        )
        return conv

    def forward(self, data):
        #encoder
        x = Variable(data['left_img'])
        conv1 = self.conv_pre(x)
        bn1 = self.bn1(conv1)
        relu = self.relu(bn1)
        pool1 = self.maxpool(relu)

        conv2 = self.layer1(pool1)   #channel 64
        conv3 = self.layer2(conv2)   #128
        conv4 = self.layer3(conv3)   #256
        conv5 = self.layer4(conv4)   #512

        #skips
        skip1 = conv1    #64
        skip2 = pool1    #64
        skip3 = conv2    #64
        skip4 = conv3   #128
        skip5 = conv4   #256

        # decoder
        upconv6 = self.upconv6(conv5)
        concat6 = torch.cat((upconv6, skip5), 1)
        iconv6 = self.iconv6(concat6)

        upconv5 = self.upconv5(iconv6)
        concat5 = torch.cat((upconv5, skip4), 1)
        iconv5 = self.iconv5(concat5)

        upconv4 = self.upconv4(iconv5)
        concat4 = torch.cat((upconv4, skip3), 1)
        iconv4 = self.iconv4(concat4)
        self.disp4 = self.disp4_layer(iconv4)
        self.udisp4 = nn.functional.interpolate(self.disp4, scale_factor=2, mode='bilinear', align_corners=True)

        upconv3 = self.upconv3(iconv4)
        concat3 = torch.cat((upconv3, skip2, self.udisp4), 1)
        iconv3 = self.iconv3(concat3)
        self.disp3 = self.disp3_layer(iconv3)
        self.udisp3 = nn.functional.interpolate(self.disp3, scale_factor=2, mode='bilinear', align_corners=True)

        upconv2 = self.upconv2(iconv3)
        concat2 = torch.cat((upconv2, skip1, self.udisp3), 1)
        iconv2 = self.iconv2(concat2)
        self.disp2 = self.disp2_layer(iconv2)
        self.udisp2 = nn.functional.interpolate(self.disp2, scale_factor=2, mode='bilinear', align_corners=True)

        upconv1 = self.upconv1(iconv2)
        concat1 = torch.cat((upconv1, self.udisp2), 1)
        iconv1 = self.iconv1(concat1)
        self.disp1 = self.disp1_layer(iconv1)

        if self.mode == "test":
            return self.disp1

        disp_list = [self.disp1, self.disp2, self.disp3, self.disp4]

        # loss = self.MyLoss(disp_list, [Variable(data['left_img']), Variable(data['right_img'])])
        loss = self.MyLoss(data, disp_list)
        return loss