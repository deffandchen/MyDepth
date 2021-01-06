#====================================================
#  File Name   : Model.py
#  Author      : deffand
#  Date        : 2020/12/17
#  Description :
#====================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


class MyLoss(nn.Module):
    def __init__(self,args):
        super(MyLoss,self).__init__()
        self.mode = args.mode
        self.ssim = SSIM()
        self.ssim.to(torch.device('cuda'))
        self.args = args

    def gradient_x(self, img):
        gx = img[:, :, :, :-1] - img[:, :, :, 1:]
        return gx

    def gradient_y(self, img):
        gy = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gy

    def apply_disparity(self, img, disp):
        batch_size, _, height, width = img.size()

        # Original coordinates of pixels
        x_base = torch.linspace(0, 1, width).repeat(batch_size,
                    height, 1).type_as(img)
        y_base = torch.linspace(0, 1, height).repeat(batch_size,
                    width, 1).transpose(1, 2).type_as(img)

        # Apply shift in X direction
        x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
        flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
        # In grid_sample coordinates are assumed to be between -1 and 1
        output = F.grid_sample(img, 2*flow_field - 1, mode='bilinear',
                               padding_mode='zeros')

        return output

    def upsample_nn(self, x, ratio):
        s = x.size()
        h = int(s[2])
        w = int(s[3])
        return nn.functional.upsample(x, [h*ratio, w*ratio], mode='bilinear')

    def scale_pyramid(self, img, num_scales):
        scaled_imgs = [img]
        s = img.size()
        h = int(s[2])
        w = int(s[3])
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_imgs.append(nn.functional.interpolate(img, size=[nh, nw], mode='bilinear'))
        return scaled_imgs

    def generate_image_left(self, img, disp):
        #return bilinear_sampler_1d_h(img, -disp)
        return self.apply_disparity(img, -disp)

    def generate_image_right(self, img, disp):
        #return bilinear_sampler_1d_h(img, disp)
        return self.apply_disparity(img,disp)

    def get_disparity_smoothness(self, disp, pyramid):
        disp_gradients_x = [self.gradient_x(d) for d in disp]
        disp_gradients_y = [self.gradient_y(d) for d in disp]

        image_gradients_x = [self.gradient_x(img) for img in pyramid]
        image_gradients_y = [self.gradient_y(img) for img in pyramid]

        weights_x = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_x]
        weights_y = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_y]

        smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(4)]
        smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(4)]
        return smoothness_x + smoothness_y

    def get_smooth_loss(self,disp, pyramid):
        """Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
        """
        grad_disp_x = [torch.abs(d[:, :, :, :-1] - d[:, :, :, 1:]) for d in disp]
        grad_disp_y = [torch.abs(d[:, :, :-1, :] - d[:, :, 1:, :]) for d in disp]

        grad_img_x = [torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True) for img in pyramid]
        grad_img_y = [torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True) for img in pyramid]

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        return grad_disp_x.mean() + grad_disp_y.mean()

    def build_model(self, data):
        self.left_pyramid = self.scale_pyramid(Variable(data['left_img']), 4)
        if self.mode == 'train':
            self.right_pyramid = self.scale_pyramid(Variable(data['right_img']), 4)

    def build_outputs(self,disp_list):
        # STORE DISPARITIES
        #with tf.variable_scope('disparities'):

        self.disp_pred = [disp_list[0], disp_list[1], disp_list[2], disp_list[3]]
        self.disp_left_pred = [torch.unsqueeze(d[:, 0, :, :], 1) for d in self.disp_pred]
        self.disp_right_pred = [torch.unsqueeze(d[:, 1, :, :], 1) for d in self.disp_pred]

        if self.mode == 'test':
            return

        # GENERATE IMAGES
        #with tf.variable_scope('images'):
        self.left_pred = [self.generate_image_left(self.right_pyramid[i], self.disp_left_pred[i]) for i in range(4)]
        self.right_pred = [self.generate_image_right(self.left_pyramid[i], self.disp_right_pred[i]) for i in range(4)]

        # LR CONSISTENCY
        #with tf.variable_scope('left-right'):
        self.disp_right_to_left = [self.generate_image_left(self.disp_right_pred[i], self.disp_left_pred[i]) for i in
                                   range(4)]
        self.disp_left_to_right = [self.generate_image_right(self.disp_left_pred[i], self.disp_right_pred[i]) for i in
                                   range(4)]

        # DISPARITY SMOOTHNESS
        #with tf.variable_scope('smoothness'):
        self.disp_left_smoothness = self.get_disparity_smoothness(self.disp_left_pred, self.left_pyramid)
        self.disp_right_smoothness = self.get_disparity_smoothness(self.disp_right_pred, self.right_pyramid)

    def build_losses(self):
        #with tf.variable_scope('losses', reuse=self.reuse_variables):
        # IMAGE RECONSTRUCTION
        # L1
        self.l1_left = [torch.abs(self.left_pred[i] - self.left_pyramid[i]) for i in range(4)]
        self.l1_reconstruction_loss_left = [torch.mean(l) for l in self.l1_left]
        self.l1_right = [torch.abs(self.right_pred[i] - self.right_pyramid[i]) for i in range(4)]
        self.l1_reconstruction_loss_right = [torch.mean(l) for l in self.l1_right]


        self.left_pix_res = [self.l1_left[i] / self.l1_reconstruction_loss_left[i] for i in range(4)]
        self.right_pix_res = [self.l1_right[i] / self.l1_reconstruction_loss_right[i] for i in range(4)]

        # SSIM
        self.ssim_left = [self.ssim(self.left_pred[i], self.left_pyramid[i]).mean(1, True) for i in range(4)]
        self.ssim_loss_left = [torch.mean(s) for s in self.ssim_left]
        self.ssim_right = [self.ssim(self.right_pred[i], self.right_pyramid[i]).mean(1, True) for i in range(4)]
        self.ssim_loss_right = [torch.mean(s) for s in self.ssim_right]

        # WEIGTHED SUM
        self.image_loss_right = [
            self.args.alpha_image_loss * self.ssim_loss_right[i] + (1 - self.args.alpha_image_loss) *
            self.l1_reconstruction_loss_right[i] for i in range(4)]
        self.image_loss_left = [
            self.args.alpha_image_loss * self.ssim_loss_left[i] + (1 - self.args.alpha_image_loss) *
            self.l1_reconstruction_loss_left[i] for i in range(4)]
        self.image_loss = np.sum(self.image_loss_left + self.image_loss_right,axis=0)

        # DISPARITY SMOOTHNESS
        self.disp_left_loss = [torch.mean(torch.abs(self.disp_left_smoothness[i])) / 2 ** i for i in range(4)]
        self.disp_right_loss = [torch.mean(torch.abs(self.disp_right_smoothness[i])) / 2 ** i for i in range(4)]
        self.disp_gradient_loss = np.sum(self.disp_left_loss + self.disp_right_loss,axis=0)

        # LR CONSISTENCY
        self.lr_left_loss = [torch.mean(torch.abs(self.disp_right_to_left[i] - self.disp_left_pred[i])) for i in
                             range(4)]
        self.lr_right_loss = [torch.mean(torch.abs(self.disp_left_to_right[i] - self.disp_right_pred[i])) for i in
                              range(4)]
        self.lr_loss = np.sum(self.lr_left_loss + self.lr_right_loss,axis=0)

        # TOTAL LOSS
        self.total_loss = self.image_loss + self.args.disp_gradient_loss_weight * self.disp_gradient_loss + self.args.lr_loss_weight * self.lr_loss


    def forward(self,data,disp_list):
        self.build_model(data)
        self.build_outputs(disp_list)
        self.build_losses()
        return self.total_loss


class MonodepthLoss(nn.modules.Module):
    def __init__(self, n=4, SSIM_w=0.85, disp_gradient_w=1.0, lr_w=1.0):
        super(MonodepthLoss, self).__init__()
        self.SSIM_w = SSIM_w
        self.disp_gradient_w = disp_gradient_w
        self.lr_w = lr_w
        self.n = n

    def scale_pyramid(self, img, num_scales):
        scaled_imgs = [img]
        s = img.size()
        h = s[2]
        w = s[3]
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_imgs.append(nn.functional.interpolate(img,
                               size=[nh, nw], mode='bilinear',
                               align_corners=True))
        return scaled_imgs

    def gradient_x(self, img):
        # Pad input to keep output size consistent
        img = F.pad(img, (0, 1, 0, 0), mode="replicate")
        gx = img[:, :, :, :-1] - img[:, :, :, 1:]  # NCHW
        return gx

    def gradient_y(self, img):
        # Pad input to keep output size consistent
        img = F.pad(img, (0, 0, 0, 1), mode="replicate")
        gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
        return gy

    def apply_disparity(self, img, disp):
        batch_size, _, height, width = img.size()

        # Original coordinates of pixels
        x_base = torch.linspace(0, 1, width).repeat(batch_size,
                    height, 1).type_as(img)
        y_base = torch.linspace(0, 1, height).repeat(batch_size,
                    width, 1).transpose(1, 2).type_as(img)

        # Apply shift in X direction
        x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
        flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
        # In grid_sample coordinates are assumed to be between -1 and 1
        output = F.grid_sample(img, 2*flow_field - 1, mode='bilinear',
                               padding_mode='zeros')

        return output

    def generate_image_left(self, img, disp):
        return self.apply_disparity(img, -disp)

    def generate_image_right(self, img, disp):
        return self.apply_disparity(img, disp)

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = nn.AvgPool2d(3, 1)(x)
        mu_y = nn.AvgPool2d(3, 1)(y)
        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
        sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
        sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

        SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
        SSIM = SSIM_n / SSIM_d

        return torch.clamp((1 - SSIM) / 2, 0, 1)

    def disp_smoothness(self, disp, pyramid):
        disp_gradients_x = [self.gradient_x(d) for d in disp]
        disp_gradients_y = [self.gradient_y(d) for d in disp]

        image_gradients_x = [self.gradient_x(img) for img in pyramid]
        image_gradients_y = [self.gradient_y(img) for img in pyramid]

        weights_x = [torch.exp(-torch.mean(torch.abs(g), 1,
                     keepdim=True)) for g in image_gradients_x]
        weights_y = [torch.exp(-torch.mean(torch.abs(g), 1,
                     keepdim=True)) for g in image_gradients_y]

        smoothness_x = [disp_gradients_x[i] * weights_x[i]
                        for i in range(self.n)]
        smoothness_y = [disp_gradients_y[i] * weights_y[i]
                        for i in range(self.n)]

        return [torch.abs(smoothness_x[i]) + torch.abs(smoothness_y[i])
                for i in range(self.n)]

    def forward(self, input, target):
        """
        Args:
            input [disp1, disp2, disp3, disp4]
            target [left, right]

        Return:
            (float): The loss
        """
        left, right = target
        left_pyramid = self.scale_pyramid(left, self.n)
        right_pyramid = self.scale_pyramid(right, self.n)

        # Prepare disparities
        disp_left_est = [d[:, 0, :, :].unsqueeze(1) for d in input]
        disp_right_est = [d[:, 1, :, :].unsqueeze(1) for d in input]

        self.disp_left_est = disp_left_est
        self.disp_right_est = disp_right_est
        # Generate images
        left_est = [self.generate_image_left(right_pyramid[i],
                    disp_left_est[i]) for i in range(self.n)]
        right_est = [self.generate_image_right(left_pyramid[i],
                     disp_right_est[i]) for i in range(self.n)]
        self.left_est = left_est
        self.right_est = right_est

        # L-R Consistency
        right_left_disp = [self.generate_image_left(disp_right_est[i],
                           disp_left_est[i]) for i in range(self.n)]
        left_right_disp = [self.generate_image_right(disp_left_est[i],
                           disp_right_est[i]) for i in range(self.n)]

        # Disparities smoothness
        disp_left_smoothness = self.disp_smoothness(disp_left_est,
                                                    left_pyramid)
        disp_right_smoothness = self.disp_smoothness(disp_right_est,
                                                     right_pyramid)

        # L1
        l1_left = [torch.mean(torch.abs(left_est[i] - left_pyramid[i]))
                   for i in range(self.n)]
        l1_right = [torch.mean(torch.abs(right_est[i]
                    - right_pyramid[i])) for i in range(self.n)]

        # SSIM
        ssim_left = [torch.mean(self.SSIM(left_est[i],
                     left_pyramid[i])) for i in range(self.n)]
        ssim_right = [torch.mean(self.SSIM(right_est[i],
                      right_pyramid[i])) for i in range(self.n)]

        image_loss_left = [self.SSIM_w * ssim_left[i]
                           + (1 - self.SSIM_w) * l1_left[i]
                           for i in range(self.n)]
        image_loss_right = [self.SSIM_w * ssim_right[i]
                            + (1 - self.SSIM_w) * l1_right[i]
                            for i in range(self.n)]
        image_loss = sum(image_loss_left + image_loss_right)

        # L-R Consistency
        lr_left_loss = [torch.mean(torch.abs(right_left_disp[i]
                        - disp_left_est[i])) for i in range(self.n)]
        lr_right_loss = [torch.mean(torch.abs(left_right_disp[i]
                         - disp_right_est[i])) for i in range(self.n)]
        lr_loss = sum(lr_left_loss + lr_right_loss)

        # Disparities smoothness
        disp_left_loss = [torch.mean(torch.abs(
                          disp_left_smoothness[i])) / 2 ** i
                          for i in range(self.n)]
        disp_right_loss = [torch.mean(torch.abs(
                           disp_right_smoothness[i])) / 2 ** i
                           for i in range(self.n)]
        disp_gradient_loss = sum(disp_left_loss + disp_right_loss)

        loss = image_loss + self.disp_gradient_w * disp_gradient_loss\
               + self.lr_w * lr_loss
        self.image_loss = image_loss
        self.disp_gradient_loss = disp_gradient_loss
        self.lr_loss = lr_loss
        return loss



def apply_disparity(input_images, x_offset, wrap_mode='border', tensor_type = 'torch.cuda.FloatTensor'):
    num_batch, num_channels, height, width = input_images.size()

    # Handle both texture border types
    edge_size = 0
    if wrap_mode == 'border':
        edge_size = 1
        # Pad last and second-to-last dimensions by 1 from both sides
        input_images = F.pad(input_images, (1, 1, 1, 1))
    elif wrap_mode == 'edge':
        edge_size = 0
    else:
        return None

    # Put channels to slowest dimension and flatten batch with respect to others
    input_images = input_images.permute(1, 0, 2, 3).contiguous()
    im_flat = input_images.view(num_channels, -1)

    # Create meshgrid for pixel indicies (PyTorch doesn't have dedicated
    # meshgrid function)
    x = torch.linspace(0, width - 1, width).repeat(height, 1).type(tensor_type).to(device = torch.device("cuda" ))
    y = torch.linspace(0, height - 1, height).repeat(width, 1).transpose(0, 1).type(tensor_type).to(device = torch.device("cuda" ))
    # Take padding into account
    x = x + edge_size
    y = y + edge_size
    # Flatten and repeat for each image in the batch
    x = x.view(-1).repeat(1, num_batch)
    y = y.view(-1).repeat(1, num_batch)

    # Now we want to sample pixels with indicies shifted by disparity in X direction
    # For that we convert disparity from % to pixels and add to X indicies
    x = x + x_offset.contiguous().view(-1) * width
    # Make sure we don't go outside of image
    x = torch.clamp(x, 0.0, width - 1 + 2 * edge_size)
    # Round disparity to sample from integer-valued pixel grid
    y0 = torch.floor(y)
    # In X direction round both down and up to apply linear interpolation
    # between them later
    x0 = torch.floor(x)
    x1 = x0 + 1
    # After rounding up we might go outside the image boundaries again
    x1 = x1.clamp(max=(width - 1 + 2 * edge_size))

    # Calculate indices to draw from flattened version of image batch
    dim2 = (width + 2 * edge_size)
    dim1 = (width + 2 * edge_size) * (height + 2 * edge_size)
    # Set offsets for each image in the batch
    base = dim1 * torch.arange(num_batch).type(tensor_type).to(device = torch.device("cuda" ))
    base = base.view(-1, 1).repeat(1, height * width).view(-1)
    # One pixel shift in Y  direction equals dim2 shift in flattened array
    base_y0 = base + y0 * dim2
    # Add two versions of shifts in X direction separately
    idx_l = base_y0 + x0
    idx_r = base_y0 + x1

    # Sample pixels from images
    pix_l = im_flat.gather(1, idx_l.repeat(num_channels, 1).long())
    pix_r = im_flat.gather(1, idx_r.repeat(num_channels, 1).long())

    # Apply linear interpolation to account for fractional offsets
    weight_l = x1 - x
    weight_r = x - x0
    output = weight_l * pix_l + weight_r * pix_r

    # Reshape back into image batch and permute back to (N,C,H,W) shape
    output = output.view(num_channels, num_batch, height, width).permute(1,0,2,3)

    return output

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