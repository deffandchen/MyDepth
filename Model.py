#
#
#

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class MyModel(nn.Module):
    def __init__(self,args):
        self.mode = args.mode

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

    def build_model(self, data):
        self.left_pyramid = self.scale_pyramid(Variable(data['left_img']), 4)
        if self.mode == 'train':
            self.right_pyramid = self.scale_pyramid(Variable(data['right_img']), 4)

    def build_outputs(self,disp_list):
        # STORE DISPARITIES
        #with tf.variable_scope('disparities'):

        self.disp_est = [disp_list[0], disp_list[1], disp_list[2], disp_list[3]]
        self.disp_left_est = [torch.unsqueeze(d[:,0, :, :], 1) for d in self.disp_est]
        self.disp_right_est = [torch.unsqueeze(d[:,1, :, :], 1) for d in self.disp_est]

        if self.mode == 'test':
            return

        # GENERATE IMAGES
        #with tf.variable_scope('images'):
        self.left_est = [self.generate_image_left(self.right_pyramid[i], self.disp_left_est[i]) for i in range(4)]
        self.right_est = [self.generate_image_right(self.left_pyramid[i], self.disp_right_est[i]) for i in range(4)]

        # LR CONSISTENCY
        #with tf.variable_scope('left-right'):
        self.right_to_left_disp = [self.generate_image_left(self.disp_right_est[i], self.disp_left_est[i]) for i in
                                   range(4)]
        self.left_to_right_disp = [self.generate_image_right(self.disp_left_est[i], self.disp_right_est[i]) for i in
                                   range(4)]

        # DISPARITY SMOOTHNESS
        #with tf.variable_scope('smoothness'):
        self.disp_left_smoothness = self.get_disparity_smoothness(self.disp_left_est, self.left_pyramid)
        self.disp_right_smoothness = self.get_disparity_smoothness(self.disp_right_est, self.right_pyramid)

    def build_losses(self):
        #with tf.variable_scope('losses', reuse=self.reuse_variables):
        # IMAGE RECONSTRUCTION
        # L1
        self.l1_left = [torch.abs(self.left_est[i] - self.left_pyramid[i]) for i in range(4)]
        self.l1_reconstruction_loss_left = [torch.mean(l) for l in self.l1_left]
        self.l1_right = [torch.abs(self.right_est[i] - self.right_pyramid[i]) for i in range(4)]
        self.l1_reconstruction_loss_right = [torch.mean(l) for l in self.l1_right]

        # SSIM
        self.ssim_left = [self.SSIM(self.left_est[i], self.left_pyramid[i]) for i in range(4)]
        self.ssim_loss_left = [torch.mean(s) for s in self.ssim_left]
        self.ssim_right = [self.SSIM(self.right_est[i], self.right_pyramid[i]) for i in range(4)]
        self.ssim_loss_right = [torch.mean(s) for s in self.ssim_right]

        # WEIGTHED SUM
        self.image_loss_right = [
            self.params.alpha_image_loss * self.ssim_loss_right[i] + (1 - self.params.alpha_image_loss) *
            self.l1_reconstruction_loss_right[i] for i in range(4)]
        self.image_loss_left = [
            self.params.alpha_image_loss * self.ssim_loss_left[i] + (1 - self.params.alpha_image_loss) *
            self.l1_reconstruction_loss_left[i] for i in range(4)]
        self.image_loss = np.sum(self.image_loss_left + self.image_loss_right,axis=0)

        # DISPARITY SMOOTHNESS
        self.disp_left_loss = [torch.mean(torch.abs(self.disp_left_smoothness[i])) / 2 ** i for i in range(4)]
        self.disp_right_loss = [torch.mean(torch.abs(self.disp_right_smoothness[i])) / 2 ** i for i in range(4)]
        self.disp_gradient_loss = np.sum(self.disp_left_loss + self.disp_right_loss,axis=0)

        # LR CONSISTENCY
        self.lr_left_loss = [torch.mean(torch.abs(self.right_to_left_disp[i] - self.disp_left_est[i])) for i in
                             range(4)]
        self.lr_right_loss = [torch.mean(torch.abs(self.left_to_right_disp[i] - self.disp_right_est[i])) for i in
                              range(4)]
        self.lr_loss = np.sum(self.lr_left_loss + self.lr_right_loss,axis=0)

        # TOTAL LOSS
        self.total_loss = self.image_loss + self.params.disp_gradient_loss_weight * self.disp_gradient_loss + self.params.lr_loss_weight * self.lr_loss


    def forward(self,data,disp_list):
        self.build_model(data)
        self.build_outputs(disp_list)
        self.build_losses()
        return self.total_loss



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