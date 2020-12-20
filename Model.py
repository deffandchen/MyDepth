#
#
#

import numpy as np
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        self.name = "image proj"
        #self.disp_list = Net.MyNet()

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

        return self.total_loss

    def forward(self,disp_list):
        self.build_outputs(disp_list)
        self.build_losses()
