import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class MyNet(nn.Module):
    def __init__(self, params, mode, left, right, reuse_variables=None, model_index=0):
        self.params = params
        self.mode = mode
        self.left = left
        self.right = right
        self.model_collection = ['model_' + str(model_index)]

        self.reuse_variables = reuse_variables

        self.build_model()
        self.build_outputs()

        if self.mode == 'test':
            return

        self.build_losses()
        self.build_summaries()

    def gradient_x(self, img):
        gx = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gx

    def gradient_y(self, img):
        gy = img[:, :-1, :, :] - img[:, 1:, :, :]
        return gy

    def upsample_nn(self, x, ratio):
        s = x.size()
        h = int(s[2])
        w = int(s[3])
        return nn.functional.upsample(x, [h*ratio, w*ratio], mode='nearest')

    def scale_pyramid(self, img, num_scales):
        scaled_imgs = [img]
        s = img.size()
        h = int(s[2])
        w = int(s[3])
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_imgs.append(nn.functional.upsample(img, [nh, nw], mode='nearest'))
        return scaled_imgs

    def generate_image_left(self, img, disp):
        return bilinear_sampler_1d_h(img, -disp)

    def generate_image_right(self, img, disp):
        return bilinear_sampler_1d_h(img, disp)

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = nn.functional.avg_pool2d(x, 3, 1, 'VALID')
        mu_y = nn.functional.avg_pool2d(y, 3, 1, 'VALID')

        sigma_x = nn.functional.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
        sigma_y = nn.functional.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2
        sigma_xy = nn.functional.avg_pool2d(x * y, 3, 1, 'VALID') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return torch.clamp((1 - SSIM) / 2, 0, 1)

    def get_disparity_smoothness(self, disp, pyramid):
        disp_gradients_x = [self.gradient_x(d) for d in disp]
        disp_gradients_y = [self.gradient_y(d) for d in disp]

        image_gradients_x = [self.gradient_x(img) for img in pyramid]
        image_gradients_y = [self.gradient_y(img) for img in pyramid]

        weights_x = [torch.exp(-torch.mean(torch.abs(g), 3, keepdim=True)) for g in image_gradients_x]
        weights_y = [torch.exp(-torch.mean(torch.abs(g), 3, keepdim=True)) for g in image_gradients_y]

        smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(4)]
        smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(4)]
        return smoothness_x + smoothness_y

    def get_disp(self, x):
        disp = 0.3 * self.conv(x, 2, 3, 1, nn.functional.sigmoid)
        return disp

    def conv(self, x, num_out_layers, kernel_size, stride):
        #p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        #p_x = torch.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return nn.Conv2d(x, num_out_layers, kernel_size, stride)

    def conv_block(self, x, num_out_layers, kernel_size):
        conv1 = self.conv(x, num_out_layers, kernel_size, 1)
        conv2 = self.conv(conv1, num_out_layers, kernel_size, 2)
        return conv2

    def maxpool(self, x, kernel_size):
        #p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        #p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return nn.MaxPool2d(x, kernel_size)

    def resconv(self, x, num_layers, stride):
        do_proj = x.size()[1] != num_layers or stride == 2
        shortcut = []
        conv1 = self.conv(x, num_layers, 1, 1)
        conv2 = self.conv(conv1, num_layers, 3, stride)
        conv3 = self.conv(conv2, 4 * num_layers, 1, 1)
        if do_proj:
            shortcut = self.conv(x, 4 * num_layers, 1, stride)
        else:
            shortcut = x
        return nn.ELU(conv3 + shortcut)

    def resblock(self, x, num_layers, num_blocks):
        out = x
        for i in range(num_blocks - 1):
            out = self.resconv(out, num_layers, 1)
        out = self.resconv(out, num_layers, 2)
        return out

    def upconv(self, x, num_out_layers, kernel_size, scale):
        upsample = self.upsample_nn(x, scale)
        conv = self.conv(upsample, num_out_layers, kernel_size, 1)
        return conv

    #def deconv(self, x, num_out_layers, kernel_size, scale):
        #p_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
        #conv = slim.conv2d_transpose(p_x, num_out_layers, kernel_size, scale, 'SAME')
        #return conv[:, 3:-1, 3:-1, :]

    def build_resnet50(self):
        #encoder
        conv1 = self.conv(self.model_input, 64, 7, 2)  # H/2  -   64D
        pool1 = self.maxpool(conv1, 3)  # H/4  -   64D
        conv2 = self.resblock(pool1, 64, 3)  # H/8  -  256D
        conv3 = self.resblock(conv2, 128, 4)  # H/16 -  512D
        conv4 = self.resblock(conv3, 256, 6)  # H/32 - 1024D
        conv5 = self.resblock(conv4, 512, 3)  # H/64 - 2048D

        #skips
        skip1 = conv1
        skip2 = pool1
        skip3 = conv2
        skip4 = conv3
        skip5 = conv4

        # decoder
        upconv6 = self.upconv(conv5, 512, 3, 2)  # H/32
        concat6 = torch.cat([upconv6, skip5], 1)
        iconv6 = self.conv(concat6, 512, 3, 1)

        upconv5 = self.upconv(iconv6, 256, 3, 2)  # H/16
        concat5 = torch.cat([upconv5, skip4], 1)
        iconv5 = self.conv(concat5, 256, 3, 1)

        upconv4 = self.upconv(iconv5, 128, 3, 2)  # H/8
        concat4 = torch.cat([upconv4, skip3], 1)
        iconv4 = self.conv(concat4, 128, 3, 1)
        self.disp4 = self.get_disp(iconv4)
        udisp4 = self.upsample_nn(self.disp4, 2)

        upconv3 = self.upconv(iconv4, 64, 3, 2)  # H/4
        concat3 = torch.cat([upconv3, skip2, udisp4], 1)
        iconv3 = self.conv(concat3, 64, 3, 1)
        self.disp3 = self.get_disp(iconv3)
        udisp3 = self.upsample_nn(self.disp3, 2)

        upconv2 = self.upconv(iconv3, 32, 3, 2)  # H/2
        concat2 = torch.cat([upconv2, skip1, udisp3], 1)
        iconv2 = self.conv(concat2, 32, 3, 1)
        self.disp2 = self.get_disp(iconv2)
        udisp2 = self.upsample_nn(self.disp2, 2)

        upconv1 = self.upconv(iconv2, 16, 3, 2)  # H
        concat1 = torch.cat([upconv1, udisp2], 1)
        iconv1 = self.conv(concat1, 16, 3, 1)
        self.disp1 = self.get_disp(iconv1)
        return [self.disp1, self.disp2, self.disp3, self.disp4]


    def forward(self):
        self.build_resnet50()


def bilinear_sampler_1d_h(input_images, x_offset, wrap_mode='edge', name='bilinear_sampler', **kwargs):
    def _repeat(x, n_repeats):
        rep = x.unsqueeze(1).repeat(1, n_repeats)
        return rep.view(-1)

    def _interpolate(im, x, y):

        # handle both texture border types
        _edge_size = 0
        if _wrap_mode == 'border':
            _edge_size = 1
            im = F.pad(im,(0,1,1,0), 'constant',0)
            x = x + _edge_size
            y = y + _edge_size
        elif _wrap_mode == 'edge':
            _edge_size = 0
        else:
            return None

        x = torch.clamp(x, 0.0,  _width_f - 1 + 2 * _edge_size)

        x0_f = torch.floor(x)
        y0_f = torch.floor(y)
        x1_f = x0_f + 1

        x0 = x0_f.type(torch.FloatTensor).cuda()
        y0 = y0_f.type(torch.FloatTensor).cuda()

        min_val = _width_f - 1 + 2 * _edge_size
        scalar = Variable(torch.FloatTensor([min_val]).cuda())

        x1 = torch.min(x1_f, scalar)
        x1 = x1.type(torch.FloatTensor).cuda()
        dim2 = (_width + 2 * _edge_size)
        dim1 = (_width + 2 * _edge_size) * (_height + 2 * _edge_size)
        base = Variable(_repeat(torch.arange(_num_batch) * dim1, _height * _width).cuda())

        base_y0 = base + y0 * dim2
        idx_l = base_y0 + x0
        idx_r = base_y0 + x1
        idx_l = idx_l.type(torch.cuda.LongTensor)
        idx_r = idx_r.type(torch.cuda.LongTensor)

        im_flat = im.contiguous().view(-1, _num_channels)
        pix_l = torch.gather(im_flat, 0, idx_l.repeat(_num_channels).view(-1, _num_channels))
        pix_r = torch.gather(im_flat, 0, idx_r.repeat(_num_channels).view(-1, _num_channels))

        weight_l = (x1_f - x).unsqueeze(1)
        weight_r = (x - x0_f).unsqueeze(1)

        return weight_l * pix_l + weight_r * pix_r

    def _transform(input_images, x_offset):

        a = Variable(torch.linspace(0.0, _width_f -1.0, _width).cuda())
        b = Variable(torch.linspace(0.0, _height_f -1.0, _height).cuda())

        x_t = a.repeat(_height)
        y_t = b.repeat(_width,1).t().contiguous().view(-1)

        x_t_flat = x_t.repeat(_num_batch, 1)
        y_t_flat = y_t.repeat(_num_batch, 1)

        x_t_flat = x_t_flat.view(-1)
        y_t_flat = y_t_flat.view(-1)

        x_t_flat = x_t_flat + x_offset.contiguous().view(-1) * _width_f

        input_transformed = _interpolate(input_images, x_t_flat, y_t_flat)

        output = input_transformed.view(_num_batch, _num_channels, _height, _width)

        return output

    _num_batch    = input_images.size(0)
    _num_channels = input_images.size(1)
    _height       = input_images.size(2)
    _width        = input_images.size(3)

    _height_f = float(_height)
    _width_f = float(_width)

    _wrap_mode = wrap_mode

    output = _transform(input_images, x_offset)

    return output