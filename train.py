#====================================================
#  File Name   : train.py
#  Author      : deffand
#  Date        : 2020/12/17
#  Description :
#====================================================

import argparse
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim

from dataset import StereoDataset
from Net import BasicBlock,MyNet
from Model import MyLoss

parser = argparse.ArgumentParser(description='Mydepth PyTorch implementation.')

parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
parser.add_argument('--model_name',                type=str,   help='model name', default='monodepth')
parser.add_argument('--encoder',                   type=str,   help='type of encoder, vgg or resnet50', default='vgg')
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti, or cityscapes', default='kitti')
parser.add_argument('--data_path',                 type=str,   help='path to the data', required=True)
parser.add_argument('--filename',            type=str,   help='path to the filenames text file', required=True)
parser.add_argument('--input_height',              type=int,   help='input height', default=256)
parser.add_argument('--input_width',               type=int,   help='input width', default=512)
parser.add_argument('--batch_size',                type=int,   help='batch size', default=8)
parser.add_argument('--start_epoch',                type=int,   help='start epoch', default=0)
parser.add_argument('--epochs',                type=int,   help='number of epochs', default=50)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--lr_loss_weight',            type=float, help='left-right consistency weight', default=1.0)
parser.add_argument('--alpha_image_loss',          type=float, help='weight between SSIM and L1 in the image loss', default=0.85)
parser.add_argument('--disp_gradient_loss_weight', type=float, help='disparity smoothness weigth', default=0.1)
parser.add_argument('--do_stereo',                             help='if set, will train the stereo model', action='store_true')
parser.add_argument('--wrap_mode',                 type=str,   help='bilinear sampler wrap mode, edge or border', default='border')
parser.add_argument('--num_gpus',                  type=int,   help='number of GPUs to use for training', default=1)
parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=8)
parser.add_argument('--output_directory',          type=str,   help='output directory for test disparities, if empty outputs to checkpoint folder', default='')
parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a specific checkpoint to load', default='')
parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')
parser.add_argument('--full_summary',                          help='if set, will keep more data for each summary. Warning: the file can become very large', action='store_true')

args = parser.parse_args()

#TODO：
def train():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_data = StereoDataset(args)  # create dataloader
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)  # , num_workers=1)
    dataset_size = len(train_data)
    print('#training images: %d' % dataset_size)

    net = MyNet(args.mode,BasicBlock)
    net.to(device)
    if args.checkpoint_path != '':
        state_dict = torch.load(args.checkpoint_path)
        net.load_state_dict(state_dict['net'])

    loss_func = MyLoss(args)
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

    for epoch in range(args.start_epoch,args.epochs):
        loss_step = 0.0
        for i, data in enumerate(train_loader):
            left = Variable(data['left_img'])
            optimizer.zero_grad()
            out = net(left)
            loss = loss_func(data, out)
            loss.backward()
            optimizer.step()
            loss_step += loss.data[0]

            if i % 100 == 0:
                print("[%d / %d, %5d]  loss: %.3f" % (epoch, args.epochs, i, loss_step))
                loss_step = 0.0

        if (epoch+1) % 10000 == 0:
            torch.save({"net": net.state_dict()},args.checkpoint_path + "model_"+str(epoch+1)+".pt")

if __name__ == '__main__':
    train()

