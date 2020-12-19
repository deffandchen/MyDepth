import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class mono_net(nn.Module): 	# vgg version
    def __init__(self, input_nc, output_nc):
        super(mono_net, self).__init__()

        self.output_nc = output_nc

        self.downconv_1 = self.conv_down_block(input_nc,32,7)
        self.downconv_2 = self.conv_down_block(32,64,5)
        self.downconv_3 = self.conv_down_block(64,128,3)
        self.downconv_4 = self.conv_down_block(128,256,3)
        self.downconv_5 = self.conv_down_block(256,512,3)
        self.downconv_6 = self.conv_down_block(512,512,3)
        self.downconv_7 = self.conv_down_block(512,512,3)

        self.upconv_7 = self.conv_up_block(512,512)
        self.upconv_6 = self.conv_up_block(512,512)
        self.upconv_5 = self.conv_up_block(512,256)
        self.upconv_4 = self.conv_up_block(256,128)
        self.upconv_3 = self.conv_up_block(128,64)
        self.upconv_2 = self.conv_up_block(64,32)
        self.upconv_1 = self.conv_up_block(32,16)

        self.conv_7 = self.conv_block(1024,512)
        self.conv_6 = self.conv_block(1024,512)
        self.conv_5 = self.conv_block(512,256)
        self.conv_4 = self.conv_block(256,128)
        self.conv_3 = self.conv_block(130,64)
        self.conv_2 = self.conv_block(66,32)
        self.conv_1 = self.conv_block(18,16)

        self.get_disp4 = self.disp_block(128)
        self.get_disp3 = self.disp_block(64)
        self.get_disp2 = self.disp_block(32)
        self.get_disp1 = self.disp_block(16)

    def conv_down_block(self, in_dim, out_dim, kernal):

        conv_down_block = []
        conv_down_block += [nn.Conv2d(in_dim, out_dim, kernel_size=kernal, stride=1, padding=int((kernal-1)/2)),nn.BatchNorm2d(out_dim),nn.ELU()]      # h,w -> h,w
        conv_down_block += [nn.Conv2d(out_dim, out_dim, kernel_size=kernal, stride=2, padding=int((kernal-1)/2)), nn.BatchNorm2d(out_dim), nn.ELU()]   # h,w -> h/2,w/2

        return nn.Sequential(*conv_down_block)

    def conv_up_block(self, in_dim, out_dim):
        conv_up_block = []
        conv_up_block += [nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),nn.ELU()]      # h,w -> h,w

        return nn.Sequential(*conv_up_block)

    def conv_block(self, in_dim, out_dim):
        conv_up_block = []
        conv_up_block += [nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),nn.ELU()]      # h,w -> h,w

        return nn.Sequential(*conv_up_block)

    def disp_block(self, in_dim):
        disp_block = []
        disp_block += [nn.Conv2d(in_dim, self.output_nc, kernel_size=3, stride=1, padding=1),nn.Sigmoid()]  # h,w -> h,w

        return nn.Sequential(*disp_block)

    def upsample_(self, disp, ratio):
        s = disp.size()
        h = int(s[2])
        w = int(s[3])
        nh = h * ratio
        nw = w * ratio
        temp = nn.functional.upsample(disp, [nh, nw], mode='nearest')

        return temp

    def forward(self, x):
                                # 3x256x512
        conv_1 = self.downconv_1(x) # 32x128x256
        conv_2 = self.downconv_2(conv_1) # 64x64x128
        conv_3 = self.downconv_3(conv_2) # 128x32x64
        conv_4 = self.downconv_4(conv_3) # 256x16x32
        conv_5 = self.downconv_5(conv_4) # 512x8x16
        conv_6 = self.downconv_6(conv_5) # 512x4x8
        conv_7 = self.downconv_7(conv_6) # 512x2x4

        conv7_up = self.upsample_(conv_7, 2) # 512x4x8
        upconv_7 = self.upconv_7(conv7_up) # 512x4x8
        concat_7 = torch.cat([upconv_7,conv_6], 1) # 1024x4x8
        iconv_7 = self.conv_7(concat_7) # 512x4x8

        iconv7_up = self.upsample_(iconv_7, 2) # 512x8x16
        upconv_6 = self.upconv_6(iconv7_up) # 512x8x16
        concat_6 = torch.cat([upconv_6,conv_5], 1) # 1024x8x16
        iconv_6 = self.conv_6(concat_6) # 512x8x16

        iconv6_up = self.upsample_(iconv_6, 2) # 512x16x32
        upconv_5 = self.upconv_5(iconv6_up) # 256x16x32
        concat_5 = torch.cat([upconv_5,conv_4], 1) # 512x16x32
        iconv_5 = self.conv_5(concat_5) # 256x16x32

        iconv5_up = self.upsample_(iconv_5, 2) # 256x32x64
        upconv_4 = self.upconv_4(iconv5_up) # 128x32x64
        concat_4 = torch.cat([upconv_4,conv_3], 1) # 256x32x64
        iconv_4 = self.conv_4(concat_4) # 128x32x64
        self.disp4 = 0.3 * self.get_disp4(iconv_4) # 2x32x64
        udisp4 = self.upsample_(self.disp4, 2) # 2x64x128

        iconv4_up = self.upsample_(iconv_4, 2) # 128x64x128
        upconv_3 = self.upconv_3(iconv4_up) # 64x64x128
        concat_3 = torch.cat([upconv_3,conv_2,udisp4], 1) # 130x64x128
        iconv_3 = self.conv_3(concat_3) # 64x64x128
        self.disp3 = 0.3 * self.get_disp3(iconv_3) # 2x64x128
        udisp3 = self.upsample_(self.disp3, 2) # 2x128x256

        iconv3_up = self.upsample_(iconv_3, 2) # 64x128x256
        upconv_2 = self.upconv_2(iconv3_up) # 32x128x256
        concat_2 = torch.cat([upconv_2,conv_1,udisp3], 1) # 66x128x256
        iconv_2 = self.conv_2(concat_2) # 32x128x256
        self.disp2 = 0.3 * self.get_disp2(iconv_2) # 2x128x256
        udisp2 = self.upsample_(self.disp2, 2) # 2x256x512

        iconv2_up = self.upsample_(iconv_2, 2) # 32x256x512
        upconv_1 = self.upconv_1(iconv2_up) # 16x256x512
        concat_1 = torch.cat([upconv_1, udisp2], 1) # 18x256x512
        iconv_1 = self.conv_1(concat_1) # 16x256x512
        self.disp1 = 0.3 * self.get_disp1(iconv_1) # 2x256x512

        return [self.disp1, self.disp2, self.disp3, self.disp4]


class Mymodel(nn.Module):
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

    def deconv(self, x, num_out_layers, kernel_size, scale):
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

    def build_outputs(self):
        # STORE DISPARITIES
        #with tf.variable_scope('disparities'):
        self.disp_est = [self.disp1, self.disp2, self.disp3, self.disp4]
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

    def forward(self):
        self.build_resnet50()


class MonodepthModel(object):
    """monodepth model"""

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
        s = tf.shape(x)
        h = s[1]
        w = s[2]
        return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])

    def scale_pyramid(self, img, num_scales):
        scaled_imgs = [img]
        s = tf.shape(img)
        h = s[1]
        w = s[2]
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_imgs.append(tf.image.resize_area(img, [nh, nw]))
        return scaled_imgs

    def generate_image_left(self, img, disp):
        return bilinear_sampler_1d_h(img, -disp)

    def generate_image_right(self, img, disp):
        return bilinear_sampler_1d_h(img, disp)

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')
        mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')

        sigma_x = slim.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
        sigma_y = slim.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y, 3, 1, 'VALID') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

    def get_disparity_smoothness(self, disp, pyramid):
        disp_gradients_x = [self.gradient_x(d) for d in disp]
        disp_gradients_y = [self.gradient_y(d) for d in disp]

        image_gradients_x = [self.gradient_x(img) for img in pyramid]
        image_gradients_y = [self.gradient_y(img) for img in pyramid]

        weights_x = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_x]
        weights_y = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_y]

        smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(4)]
        smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(4)]
        return smoothness_x + smoothness_y

    def get_disp(self, x):
        disp = 0.3 * self.conv(x, 2, 3, 1, tf.nn.sigmoid)
        return disp

    def conv(self, x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.elu):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn)

    def conv_block(self, x, num_out_layers, kernel_size):
        conv1 = self.conv(x, num_out_layers, kernel_size, 1)
        conv2 = self.conv(conv1, num_out_layers, kernel_size, 2)
        return conv2

    def maxpool(self, x, kernel_size):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.max_pool2d(p_x, kernel_size)

    def resconv(self, x, num_layers, stride):
        do_proj = tf.shape(x)[3] != num_layers or stride == 2
        shortcut = []
        conv1 = self.conv(x, num_layers, 1, 1)
        conv2 = self.conv(conv1, num_layers, 3, stride)
        conv3 = self.conv(conv2, 4 * num_layers, 1, 1, None)
        if do_proj:
            shortcut = self.conv(x, 4 * num_layers, 1, stride, None)
        else:
            shortcut = x
        return tf.nn.elu(conv3 + shortcut)

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

    def deconv(self, x, num_out_layers, kernel_size, scale):
        p_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
        conv = slim.conv2d_transpose(p_x, num_out_layers, kernel_size, scale, 'SAME')
        return conv[:, 3:-1, 3:-1, :]

    def build_vgg(self):
        # set convenience functions
        conv = self.conv
        if self.params.use_deconv:
            upconv = self.deconv
        else:
            upconv = self.upconv

        with tf.variable_scope('encoder'):
            conv1 = self.conv_block(self.model_input, 32, 7)  # H/2
            conv2 = self.conv_block(conv1, 64, 5)  # H/4
            conv3 = self.conv_block(conv2, 128, 3)  # H/8
            conv4 = self.conv_block(conv3, 256, 3)  # H/16
            conv5 = self.conv_block(conv4, 512, 3)  # H/32
            conv6 = self.conv_block(conv5, 512, 3)  # H/64
            conv7 = self.conv_block(conv6, 512, 3)  # H/128

        with tf.variable_scope('skips'):
            skip1 = conv1
            skip2 = conv2
            skip3 = conv3
            skip4 = conv4
            skip5 = conv5
            skip6 = conv6

        with tf.variable_scope('decoder'):
            upconv7 = upconv(conv7, 512, 3, 2)  # H/64
            concat7 = tf.concat([upconv7, skip6], 3)
            iconv7 = conv(concat7, 512, 3, 1)

            upconv6 = upconv(iconv7, 512, 3, 2)  # H/32
            concat6 = tf.concat([upconv6, skip5], 3)
            iconv6 = conv(concat6, 512, 3, 1)

            upconv5 = upconv(iconv6, 256, 3, 2)  # H/16
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5 = conv(concat5, 256, 3, 1)

            upconv4 = upconv(iconv5, 128, 3, 2)  # H/8
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4 = conv(concat4, 128, 3, 1)
            self.disp4 = self.get_disp(iconv4)
            udisp4 = self.upsample_nn(self.disp4, 2)

            upconv3 = upconv(iconv4, 64, 3, 2)  # H/4
            concat3 = tf.concat([upconv3, skip2, udisp4], 3)
            iconv3 = conv(concat3, 64, 3, 1)
            self.disp3 = self.get_disp(iconv3)
            udisp3 = self.upsample_nn(self.disp3, 2)

            upconv2 = upconv(iconv3, 32, 3, 2)  # H/2
            concat2 = tf.concat([upconv2, skip1, udisp3], 3)
            iconv2 = conv(concat2, 32, 3, 1)
            self.disp2 = self.get_disp(iconv2)
            udisp2 = self.upsample_nn(self.disp2, 2)

            upconv1 = upconv(iconv2, 16, 3, 2)  # H
            concat1 = tf.concat([upconv1, udisp2], 3)
            iconv1 = conv(concat1, 16, 3, 1)
            self.disp1 = self.get_disp(iconv1)

    def build_resnet50(self):
        # set convenience functions
        conv = self.conv
        if self.params.use_deconv:
            upconv = self.deconv
        else:
            upconv = self.upconv

        with tf.variable_scope('encoder'):
            conv1 = conv(self.model_input, 64, 7, 2)  # H/2  -   64D
            pool1 = self.maxpool(conv1, 3)  # H/4  -   64D
            conv2 = self.resblock(pool1, 64, 3)  # H/8  -  256D
            conv3 = self.resblock(conv2, 128, 4)  # H/16 -  512D
            conv4 = self.resblock(conv3, 256, 6)  # H/32 - 1024D
            conv5 = self.resblock(conv4, 512, 3)  # H/64 - 2048D

        with tf.variable_scope('skips'):
            skip1 = conv1
            skip2 = pool1
            skip3 = conv2
            skip4 = conv3
            skip5 = conv4

        # DECODING
        with tf.variable_scope('decoder'):
            upconv6 = upconv(conv5, 512, 3, 2)  # H/32
            concat6 = tf.concat([upconv6, skip5], 3)
            iconv6 = conv(concat6, 512, 3, 1)

            upconv5 = upconv(iconv6, 256, 3, 2)  # H/16
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5 = conv(concat5, 256, 3, 1)

            upconv4 = upconv(iconv5, 128, 3, 2)  # H/8
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4 = conv(concat4, 128, 3, 1)
            self.disp4 = self.get_disp(iconv4)
            udisp4 = self.upsample_nn(self.disp4, 2)

            upconv3 = upconv(iconv4, 64, 3, 2)  # H/4
            concat3 = tf.concat([upconv3, skip2, udisp4], 3)
            iconv3 = conv(concat3, 64, 3, 1)
            self.disp3 = self.get_disp(iconv3)
            udisp3 = self.upsample_nn(self.disp3, 2)

            upconv2 = upconv(iconv3, 32, 3, 2)  # H/2
            concat2 = tf.concat([upconv2, skip1, udisp3], 3)
            iconv2 = conv(concat2, 32, 3, 1)
            self.disp2 = self.get_disp(iconv2)
            udisp2 = self.upsample_nn(self.disp2, 2)

            upconv1 = upconv(iconv2, 16, 3, 2)  # H
            concat1 = tf.concat([upconv1, udisp2], 3)
            iconv1 = conv(concat1, 16, 3, 1)
            self.disp1 = self.get_disp(iconv1)

    def build_model(self):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.elu):
            with tf.variable_scope('model', reuse=self.reuse_variables):

                self.left_pyramid = self.scale_pyramid(self.left, 4)
                if self.mode == 'train':
                    self.right_pyramid = self.scale_pyramid(self.right, 4)

                if self.params.do_stereo:
                    self.model_input = tf.concat([self.left, self.right], 3)
                else:
                    self.model_input = self.left

                # build model
                if self.params.encoder == 'vgg':
                    self.build_vgg()
                elif self.params.encoder == 'resnet50':
                    self.build_resnet50()
                else:
                    return None

    def build_outputs(self):
        # STORE DISPARITIES
        with tf.variable_scope('disparities'):
            self.disp_est = [self.disp1, self.disp2, self.disp3, self.disp4]
            self.disp_left_est = [tf.expand_dims(d[:, :, :, 0], 3) for d in self.disp_est]
            self.disp_right_est = [tf.expand_dims(d[:, :, :, 1], 3) for d in self.disp_est]

        if self.mode == 'test':
            return

        # GENERATE IMAGES
        with tf.variable_scope('images'):
            self.left_est = [self.generate_image_left(self.right_pyramid[i], self.disp_left_est[i]) for i in range(4)]
            self.right_est = [self.generate_image_right(self.left_pyramid[i], self.disp_right_est[i]) for i in range(4)]

        # LR CONSISTENCY
        with tf.variable_scope('left-right'):
            self.right_to_left_disp = [self.generate_image_left(self.disp_right_est[i], self.disp_left_est[i]) for i in
                                       range(4)]
            self.left_to_right_disp = [self.generate_image_right(self.disp_left_est[i], self.disp_right_est[i]) for i in
                                       range(4)]

        # DISPARITY SMOOTHNESS
        with tf.variable_scope('smoothness'):
            self.disp_left_smoothness = self.get_disparity_smoothness(self.disp_left_est, self.left_pyramid)
            self.disp_right_smoothness = self.get_disparity_smoothness(self.disp_right_est, self.right_pyramid)

    def build_losses(self):
        with tf.variable_scope('losses', reuse=self.reuse_variables):
            # IMAGE RECONSTRUCTION
            # L1
            self.l1_left = [tf.abs(self.left_est[i] - self.left_pyramid[i]) for i in range(4)]
            self.l1_reconstruction_loss_left = [tf.reduce_mean(l) for l in self.l1_left]
            self.l1_right = [tf.abs(self.right_est[i] - self.right_pyramid[i]) for i in range(4)]
            self.l1_reconstruction_loss_right = [tf.reduce_mean(l) for l in self.l1_right]

            # SSIM
            self.ssim_left = [self.SSIM(self.left_est[i], self.left_pyramid[i]) for i in range(4)]
            self.ssim_loss_left = [tf.reduce_mean(s) for s in self.ssim_left]
            self.ssim_right = [self.SSIM(self.right_est[i], self.right_pyramid[i]) for i in range(4)]
            self.ssim_loss_right = [tf.reduce_mean(s) for s in self.ssim_right]

            # WEIGTHED SUM
            self.image_loss_right = [
                self.params.alpha_image_loss * self.ssim_loss_right[i] + (1 - self.params.alpha_image_loss) *
                self.l1_reconstruction_loss_right[i] for i in range(4)]
            self.image_loss_left = [
                self.params.alpha_image_loss * self.ssim_loss_left[i] + (1 - self.params.alpha_image_loss) *
                self.l1_reconstruction_loss_left[i] for i in range(4)]
            self.image_loss = tf.add_n(self.image_loss_left + self.image_loss_right)

            # DISPARITY SMOOTHNESS
            self.disp_left_loss = [tf.reduce_mean(tf.abs(self.disp_left_smoothness[i])) / 2 ** i for i in range(4)]
            self.disp_right_loss = [tf.reduce_mean(tf.abs(self.disp_right_smoothness[i])) / 2 ** i for i in range(4)]
            self.disp_gradient_loss = tf.add_n(self.disp_left_loss + self.disp_right_loss)

            # LR CONSISTENCY
            self.lr_left_loss = [tf.reduce_mean(tf.abs(self.right_to_left_disp[i] - self.disp_left_est[i])) for i in
                                 range(4)]
            self.lr_right_loss = [tf.reduce_mean(tf.abs(self.left_to_right_disp[i] - self.disp_right_est[i])) for i in
                                  range(4)]
            self.lr_loss = tf.add_n(self.lr_left_loss + self.lr_right_loss)

            # TOTAL LOSS
            self.total_loss = self.image_loss + self.params.disp_gradient_loss_weight * self.disp_gradient_loss + self.params.lr_loss_weight * self.lr_loss

    def build_summaries(self):
        # SUMMARIES
        with tf.device('/cpu:0'):
            for i in range(4):
                tf.summary.scalar('ssim_loss_' + str(i), self.ssim_loss_left[i] + self.ssim_loss_right[i],
                                  collections=self.model_collection)
                tf.summary.scalar('l1_loss_' + str(i),
                                  self.l1_reconstruction_loss_left[i] + self.l1_reconstruction_loss_right[i],
                                  collections=self.model_collection)
                tf.summary.scalar('image_loss_' + str(i), self.image_loss_left[i] + self.image_loss_right[i],
                                  collections=self.model_collection)
                tf.summary.scalar('disp_gradient_loss_' + str(i), self.disp_left_loss[i] + self.disp_right_loss[i],
                                  collections=self.model_collection)
                tf.summary.scalar('lr_loss_' + str(i), self.lr_left_loss[i] + self.lr_right_loss[i],
                                  collections=self.model_collection)
                tf.summary.image('disp_left_est_' + str(i), self.disp_left_est[i], max_outputs=4,
                                 collections=self.model_collection)
                tf.summary.image('disp_right_est_' + str(i), self.disp_right_est[i], max_outputs=4,
                                 collections=self.model_collection)

                if self.params.full_summary:
                    tf.summary.image('left_est_' + str(i), self.left_est[i], max_outputs=4,
                                     collections=self.model_collection)
                    tf.summary.image('right_est_' + str(i), self.right_est[i], max_outputs=4,
                                     collections=self.model_collection)
                    tf.summary.image('ssim_left_' + str(i), self.ssim_left[i], max_outputs=4,
                                     collections=self.model_collection)
                    tf.summary.image('ssim_right_' + str(i), self.ssim_right[i], max_outputs=4,
                                     collections=self.model_collection)
                    tf.summary.image('l1_left_' + str(i), self.l1_left[i], max_outputs=4,
                                     collections=self.model_collection)
                    tf.summary.image('l1_right_' + str(i), self.l1_right[i], max_outputs=4,
                                     collections=self.model_collection)

            if self.params.full_summary:
                tf.summary.image('left', self.left, max_outputs=4, collections=self.model_collection)
                tf.summary.image('right', self.right, max_outputs=4, collections=self.model_collection)


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