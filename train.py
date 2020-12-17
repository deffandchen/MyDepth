
import torch
from torch.utils.data import DataLoader

from dataloader import Stereoloader

data_loader = Stereoloader(opt) # create dataloader
train_data = DataLoader(data_loader, batch_size=opt.batchsize, shuffle=True  )  # , num_workers=1)
dataset_size = len(data_loader)
print('#training images: %d' %dataset_size)