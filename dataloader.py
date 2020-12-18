#
# Author : deffand
#
# Monodepth
#

from torch.utils.data.dataset import Dataset
import numpy as np
from PIL import Image
import random
import torchvision.transforms as transforms

class Stereoloader(Dataset):
    _left = []
    _right = []

    def __init__(self, opt):
        self.opt = opt
        filename = self.opt.filename
        dataroot = self.opt.dataroot
        arrlenth = 66 + len(dataroot)
        arrlen = '|S' + str(arrlenth)
        arr = np.genfromtxt(filename, dtype=str, delimiter=' ')
        n_line = open(filename).read().count('\n')
        for line in range(n_line):
            self._left.append(dataroot + arr[line][0])
            self._right.append(dataroot + arr[line][1])

    def __getitem__(self, index):
        img1 = Image.open(self._left[index])
        params = get_params(self.opt, img1.size)

        img1 = Image.open(self._left[index]).convert('RGB')
        img2 = Image.open(self._right[index]).convert('RGB')
        # not use augument
        #arg = random.random() > 0.5
        #if arg:
        #    img1, img2 = self.augument_image_pair(img1, img2)

        transform = get_transform(self.opt, params)
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
        return len(self.__left)


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.resize_or_crop == 'resize_and_crop':
        new_h = new_w = opt.loadSize
    elif opt.resize_or_crop == 'scale_width_and_crop':
        new_w = opt.loadSize
        new_h = opt.loadSize * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.fineSize))
    y = random.randint(0, np.maximum(0, new_h - opt.fineSize))

    flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(opt, params, method=Image.BICUBIC, normalize=True):
    transform_list = []
    if 'resize' in opt.resize_or_crop:
        osize = [opt.input_height, opt.input_width]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.loadSize, method)))

    if 'crop' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.fineSize)))

    if opt.resize_or_crop == 'none':
        base = float(2 ** opt.n_downsample_global)
        if opt.netG == 'local':
            base *= (2 ** opt.n_local_enhancers)
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def normalize():
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img