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

class MyNet(nn.Module):

    def __init__(self, args, block):
        super(MyNet, self).__init__()
        self.mode = args.mode
        self.MyLoss = MonodepthLoss()
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
        self.upconv6 = self.upconv(512, 3, 1, 1)
        self.upconv5 = self.upconv(256, 3, 1, 3)
        self.upconv4 = self.upconv(128, 3, 1, 3)
        self.upconv3 = self.upconv(64, 3, 1, 3)
        self.upconv2 = self.upconv(32, 3, 1, 3)
        self.upconv1 = self.upconv(16, 3, 1, 3)

        self.conv6 = self.conv(512+256, 512, 3, 1)
        self.conv5 = self.conv(256+128, 256, 3, 1)
        self.conv4 = self.conv(128+64, 128, 3, 1)
        self.conv3 = self.conv(64+64, 64, 3, 1)
        self.conv2 = self.conv(32+64, 32, 3, 1)
        self.conv1 = self.conv(16, 16, 3, 1)

        self.get_disp4 = self.disp_block(128)
        self.get_disp3 = self.disp_block(64)
        self.get_disp2 = self.disp_block(32)
        self.get_disp1 = self.disp_block(16)

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
        upsample6 = self.upsample(conv5,2)
        upconv6 = self.upconv6(upsample6)  # H/32
        concat6 = torch.cat([upconv6, skip5], 1)   #512+256
        #self.inplanes = concat6.size()[1]
        iconv6 = self.conv6(concat6)         #512

        upsample5 = self.upsample(iconv6, 2)
        upconv5 = self.upconv5(upsample5)  # H/16
        concat5 = torch.cat([upconv5, skip4], 1)    #256+128
        iconv5 = self.conv5(concat5)          #256

        upsample4 = self.upsample(iconv5, 2)
        upconv4 = self.upconv4(upsample4)  # H/8
        concat4 = torch.cat([upconv4, skip3], 1)  #128+64
        iconv4 = self.conv4(concat4)           # 128
        self.disp4 = 0.3 * self.get_disp4(iconv4)     # 128
        udisp4 = self.upsample(self.disp4, 2)

        upsample3 = self.upsample(iconv4,2)
        upconv3 = self.upconv3(upsample3)  # H/4
        #concat3 = torch.cat([upconv3, skip2, udisp4], 1)   #64+64 + 128
        concat3 = torch.cat([upconv3, skip2], 1)  # 64+64
        iconv3 = self.conv3(concat3)                    #64
        self.disp3 = 0.3 * self.get_disp3(iconv3)            #64
        udisp3 = self.upsample(self.disp3, 2)

        upsample2 = self.upsample(iconv3,2)
        upconv2 = self.upconv2(upsample2)  # H/2   32
        concat2 = torch.cat([upconv2, skip1], 1)    # 32+64
        iconv2 = self.conv2(concat2)
        self.disp2 = 0.3 * self.get_disp2(iconv2)          #32
        udisp2 = self.upsample(self.disp2, 2)

        upsample1 = self.upsample(iconv2,2)
        upconv1 = self.upconv1(upsample1)  # H   16
        #concat1 = torch.cat([upconv1], 1) # 16+ 32
        iconv1 = self.conv1(upconv1)
        self.disp1 = 0.3 * self.get_disp1(iconv1)        #16
        disp_list =  [self.disp1, self.disp2, self.disp3, self.disp4]

        loss = self.MyLoss(disp_list,[Variable(data['left_img']),Variable(data['right_img'])])
        return loss

