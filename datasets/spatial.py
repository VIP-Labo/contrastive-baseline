# in  : original image
# out : cropped img1 (anchor)
#       cropped img2 (compete)
#       target (positive img1 - img2 : 1, negative img1 - img2 : 0)

import os
from glob import glob
import random

import numpy as np
from PIL import Image
from PIL import ImageFilter

import torch
import torch.utils.data as data
import torchvision.transforms.functional as F
from torchvision import transforms

random.seed(765)

def equally_divide_patches(img, divide_num):
    patche_size_w = int(img.size[0] / divide_num) 
    patche_size_h = int(img.size[1] / divide_num)

    patches = []
    for cnt_i, i in enumerate(range(0, img.size[1], patche_size_h)):
        if cnt_i == divide_num:
            break
        for cnt_j, j in enumerate(range(0, img.size[0], patche_size_w)):
            if cnt_j == divide_num:
                break
            box = (j, i, j+patche_size_w, i+patche_size_h)
            patches.append(img.crop(box))

    return patches

def create_pos_pair(patches):
    idx = random.randint(0, len(patches)-1)
    img1 = patches[idx]
    img2 = patches[idx]
    target = np.array([1])
    return img1, img2, target

def create_neg_pair(patches):
    idx = random.sample(range(0, len(patches)-1), k=2)
    img1 = patches[idx[0]]
    img2 = patches[idx[1]]
    target = np.array([0])
    return img1, img2, target

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class SpatialDataset(data.Dataset):
    # divide_num : 3 -> 3x3= 9 paches
    def __init__(self, data_path, crop_size, divide_num=3, aug=True):
        self.data_path = data_path
        self.im_list = sorted(glob(os.path.join(self.data_path, '*.jpg')))

        self.c_size = crop_size
        self.d_num = divide_num

        if aug:
            self.aug = transforms.Compose([
                transforms.CenterCrop(self.c_size),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip()
            ])
        else:
            self.aug = transforms.CenterCrop(self.c_size)

        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, index):
        img_path = self.im_list[index]
        img = Image.open(img_path).convert('RGB')
        patches = equally_divide_patches(img, self.d_num)

        if random.random() > 0.5:
            img1, img2, target = create_pos_pair(patches)
        else:
            img1, img2, target = create_neg_pair(patches)

        img1 = self.aug(img1)
        img2 = self.aug(img2)

        target = torch.from_numpy(target).long()

        img1 = self.trans(img1)
        img2 = self.trans(img2)

        return img1, img2, target, None