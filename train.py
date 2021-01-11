#====================================================
#  File Name   : train.py
#  Author      : deffand
#  Date        : 2020/12/17
#  Description :
#====================================================

import os
import sys
import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim

from dataset import StereoDataset
from Net import BasicBlock,MyNet
from Senet import SENet,SEBottleneck
from Model import MyLoss
from testNet import Resnet50_md

#/media/lab326/9a55ef08-6e15-4a6e-b1c5-9f20232c2f002/lab326/cdf

#os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'

parser = argparse.ArgumentParser(description='Mydepth PyTorch implementation.')

parser.add_argument('--mode',                      type=str,   help='train or test', default='test')
parser.add_argument('--model_name',                type=str,   help='model name', default='monodepth')
parser.add_argument('--encoder',                   type=str,   help='type of encoder, vgg or resnet50', default='vgg')
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti, or cityscapes', default='kitti')
parser.add_argument('--data_path',                 type=str,   help='path to the data', required=True)
parser.add_argument('--filename',                  type=str,   help='path to the filenames text file', default="utils/filenames/kitti_train_files.txt")
parser.add_argument('--input_height',              type=int,   help='input height', default=256)
parser.add_argument('--input_width',               type=int,   help='input width', default=512)
parser.add_argument('--batch_size',                type=int,   help='batch size', default=32)
parser.add_argument('--start_epoch',                type=int,   help='start epoch', default=0)
parser.add_argument('--epochs',                type=int,   help='number of epochs', default=50)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-3)
parser.add_argument('--lr_loss_weight',            type=float, help='left-right consistency weight', default=1.0)
parser.add_argument('--alpha_image_loss',          type=float, help='weight between SSIM and L1 in the image loss', default=0.85)
parser.add_argument('--disp_gradient_loss_weight', type=float, help='disparity smoothness weigth', default=0.1)
parser.add_argument('--do_stereo',                             help='if set, will train the stereo model', action='store_true')
parser.add_argument('--wrap_mode',                 type=str,   help='bilinear sampler wrap mode, edge or border', default='border')
parser.add_argument('--num_gpus',                  type=int,   help='number of GPUs to use for training', default=2)
parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=8)
parser.add_argument('--output_directory',          type=str,   help='output directory for test disparities, if empty outputs to checkpoint folder', default='')
parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a specific checkpoint to load', default="")
parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')
parser.add_argument('--full_summary',                          help='if set, will keep more data for each summary. Warning: the file can become very large', action='store_true')

args = parser.parse_args()

def post_process_disparity(disp):
    (_, h, w) = disp.shape
    l_disp = disp[0, :, :]
    r_disp = np.fliplr(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)
    (l, _) = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def adjust_learning_rate(optimizer, epoch, learning_rate):
    """Sets the learning rate to the initial LR\
        decayed by 2 every 10 epochs after 30 epoches"""

    if epoch >= 40 and epoch < 70:
        lr = learning_rate / 10.0
    elif epoch >= 70:
        lr = learning_rate / 10.0
    else:
        lr = learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

#TODO：
def train():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_data = StereoDataset(args)  # create dataloader
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)  # , num_workers=1)
    dataset_size = len(train_data)
    iter = dataset_size // args.batch_size
    print('#training images: %d' % dataset_size)

    #net = MyNet(args,BasicBlock)
    net = SENet(args,SEBottleneck)
    #net = Resnet50_md(3,args)
    if args.num_gpus > 1:
        net = torch.nn.DataParallel(net).cuda()
    else:
        net.to(device)

    net.train()

    if args.checkpoint_path != '':
        state_dict = torch.load(args.checkpoint_path)
        net.load_state_dict(state_dict['net'])

    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

    for epoch in range(args.start_epoch,args.epochs):
        adjust_learning_rate(optimizer,epoch,args.learning_rate)
        start_time = time.time()
        train_loss = 0.0
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            loss = net(data)
            loss = loss.mean()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if (i) % 100 == 0:
                print("[%d / %d, %5d]  loss: %.3f" % (epoch, args.epochs, i, loss.item()))
        print('Epoch:', epoch + 1, 'train_loss:', train_loss / iter, 'time:',
            round(time.time() - start_time, 3), 's')
        torch.save({"net": net.state_dict()},args.output_directory + "model_"+str(epoch+1)+".pt")


def test():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    val_data = StereoDataset(args)  # create dataloader
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)  # , num_workers=1)
    dataset_size = len(val_data)
    print('test images: %d' % dataset_size)

    #net = MyNet(args,BasicBlock)
    net = SENet(args, SEBottleneck)
    net = torch.nn.DataParallel(net).cuda()
    net.eval()

    if args.checkpoint_path != '':
        state_dict = torch.load(args.checkpoint_path)
        net.load_state_dict(state_dict['net'])
    else:
        print("please input checkpoint path for test!")
        sys.exit(0)

    disparities = np.zeros((dataset_size,
                            args.input_height, args.input_width),
                           dtype=np.float32)
    disparities_pp = np.zeros((dataset_size,
                               args.input_height, args.input_width),
                              dtype=np.float32)

    with torch.no_grad():
        for (i, data) in enumerate(val_loader):
            # Get the inputs
            #left = data['left_img']
            # Do a forward pass
            disps = net(data)
            disp = disps[:, 0, :, :].unsqueeze(1)
            disparities[i] = disp[0].squeeze().cpu().numpy()
            #disparities_pp[i] = post_process_disparity(disps[0][:, 0, :, :] \
            #                           .cpu().numpy())
            print("test: ",i)

    np.save(args.output_directory + '/disparities.npy', disparities)
    #np.save(args.output_directory + '/disparities_pp.npy', disparities_pp)
    print('Finished Testing')

if __name__ == '__main__':
    if args.mode == "train":
        train()
    else:
        test()

