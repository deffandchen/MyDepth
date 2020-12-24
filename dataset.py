#====================================================
#  File Name   : dataset.py
#  Author      : deffand
#  Date        : 2020/12/17
#  Description :
#====================================================

from torch.utils.data.dataset import Dataset
import numpy as np
from PIL import Image
import random
import torchvision.transforms as transforms

class StereoDataset(Dataset):
    _left = []
    _right = []

    def __init__(self, args):
        self.args = args
        filename = self.args.filename
        dataroot = self.args.data_path
        self.normalize = True

        files = np.genfromtxt(filename, dtype=str, delimiter=' ')
        #n_line = open(filename).read().count('\n')
        for f in files:
            self._left.append(dataroot + f[0])
            self._right.append(dataroot + f[1])

    def __getitem__(self, index):
        img1 = Image.open(self._left[index]).convert('RGB')
        img2 = Image.open(self._right[index]).convert('RGB')
        # not use augument
        #arg = random.random() > 0.5
        #if arg:
        #    img1, img2 = self.augument_image_pair(img1, img2)

        #transforms
        transform_list = []
        out_size = [self.args.input_height, self.args.input_width]
        transform_list.append(transforms.Resize(out_size, interpolation=Image.BICUBIC))

        transform_list += [transforms.ToTensor()]

        if self.normalize:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                    (0.5, 0.5, 0.5))]

        transform = transforms.Compose(transform_list)
        img1 = transform(img1)
        img2 = transform(img2)

        input_dict = {'left_img': img1.cuda(), 'right_img': img2.cuda()}

        return input_dict

    def augument_image_pair(self, left_image, right_image):

        left_image = np.asarray(left_image)
        right_image = np.asarray(right_image)
        # print(np.amin(left_image))

        # randomly gamma shift
        random_gamma = random.uniform(0.8, 1.2)
        left_image_aug = left_image ** random_gamma
        right_image_aug = right_image ** random_gamma

        # randomly shift brightness
        random_brightness = random.uniform(0.5, 2.0)
        left_image_aug = left_image_aug * random_brightness
        right_image_aug = right_image_aug * random_brightness

        # randomly shift color
        # random_colors = [random.uniform(0.8, 1.2), random.uniform(0.8, 1.2), random.uniform(0.8, 1.2)]
        # white = np.ones((left_image.shape[0],left_image.shape[1]))
        # color_image = np.stack([white * random_colors[i] for i in range(3)], axis=2)
        # left_image_aug  *= color_image
        # right_image_aug *= color_image

        # saturate
        # left_image_aug  = np.clip(left_image_aug,  0, 1)
        # right_image_aug = np.clip(right_image_aug, 0, 1)

        left_image_aug = Image.fromarray(np.uint8(left_image_aug))
        right_image_aug = Image.fromarray(np.uint8(right_image_aug))

        return left_image_aug, right_image_aug

    def __len__(self):
        return len(self._left)
