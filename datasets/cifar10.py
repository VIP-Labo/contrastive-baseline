from typing import Callable, Optional
import random

from PIL import Image
import numpy as np

import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10

np.random.seed(765)
random.seed(765)

class SupervisedPosNegCifar10(torch.utils.data.Dataset):
    def __init__(self, dataset, phase):
        # split by some thresholds here 80% anchors, 20% for posnegs
        lengths = [int(len(dataset)*0.8), int(len(dataset)*0.2)]
        self.anchors, self.posnegs = torch.utils.data.random_split(dataset, lengths)
        
        if phase == 'train':
            self.anchor_transform = transforms.Compose([transforms.Resize(64),
                        transforms.RandomResizedCrop(scale=(0.16, 1), ratio=(0.75, 1.33), size=64),
                        transforms.RandomHorizontalFlip(0.5),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            self.posneg_transform = transforms.Compose([transforms.Resize(64),
                        transforms.RandomResizedCrop(scale=(0.16, 1), ratio=(0.75, 1.33), size=64),
                        transforms.RandomHorizontalFlip(0.5),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        else:
            self.anchor_transform = transforms.Compose([transforms.Resize(64),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            self.posneg_transform = transforms.Compose([transforms.Resize(64),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.anchors)
        
    def __getitem__(self, index):            
        anchor, label = self.anchors[index]
        if self.anchor_transform is not None:
            anchor = self.anchor_transform(anchor)
        
        # now pair this up with an image from the same class in the second stream
        if random.random() > 0.5:
            A = np.where(np.array(self.posnegs.dataset.targets) == label)[0]
            posneg_idx = np.random.choice(A[np.in1d(A, self.posnegs.indices)])
            posneg, label = self.posnegs[np.where(self.posnegs.indices==posneg_idx)[0][0]]
            target = torch.tensor([1]).long()
        else:
            A = np.where(np.array(self.posnegs.dataset.targets) != label)[0]
            posneg_idx = np.random.choice(A[np.in1d(A, self.posnegs.indices)])
            posneg, label = self.posnegs[np.where(self.posnegs.indices==posneg_idx)[0][0]]
            target = torch.tensor([0]).long()

        if self.posneg_transform is not None:
            posneg = self.posneg_transform(posneg)

        return anchor, posneg, target, label

class PosNegCifar10(torch.utils.data.Dataset):
    def __init__(self, dataset, phase):
        # split by some thresholds here 80% anchors, 20% for posnegs
        self.dataset = dataset
        
        if phase == 'train':
            self.anchor_transform = transforms.Compose([transforms.Resize(64),
                        transforms.RandomResizedCrop(scale=(0.16, 1), ratio=(0.75, 1.33), size=64),
                        transforms.RandomHorizontalFlip(0.5),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            self.posneg_transform = transforms.Compose([transforms.Resize(64),
                        transforms.RandomResizedCrop(scale=(0.16, 1), ratio=(0.75, 1.33), size=64),
                        transforms.RandomHorizontalFlip(0.5),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        else:
            self.anchor_transform = transforms.Compose([transforms.Resize(64),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            self.posneg_transform = transforms.Compose([transforms.Resize(64),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, index):            
        anchor, label = self.dataset[index]

        # now pair this up with an image from the same class in the second stream
        if random.random() > 0.5:
            posneg = anchor
            target = torch.tensor([1]).long()
        else:
            while True:
                neg_idx = random.randint(0, len(self.dataset)-1)
                if neg_idx != index:
                    break
            posneg, label = self.dataset[neg_idx]
            target = torch.tensor([0]).long()

        if self.anchor_transform is not None:
            anchor = self.anchor_transform(anchor)

        if self.posneg_transform is not None:
            posneg = self.posneg_transform(posneg)

        return anchor, posneg, target, label

### Simple Siamese code

imagenet_mean_std = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]

class SimSiamTransform():
    def __init__(self, image_size, train, mean_std=imagenet_mean_std):
        self.train = train
        if self.train:
            image_size = 224 if image_size is None else image_size # by default simsiam use image size 224
            p_blur = 0.5 if image_size > 32 else 0 # exclude cifar
            # the paper didn't specify this, feel free to change this value
            # I use the setting from simclr which is 50% chance applying the gaussian blur
            # the 32 is prepared for cifar training where they disabled gaussian blur
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=p_blur),
                transforms.ToTensor(),
                transforms.Normalize(*mean_std)
            ])

        else:
            self.transform = transforms.Compose([
                transforms.Resize(int(image_size*(8/7)), interpolation=Image.BICUBIC), # 224 -> 256 
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(*mean_std)
            ])

    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2 


def get_simsiam_dataset(args, phase, download=True, debug_subset_size=None):
    if phase == 'train':
        train = True
        transform = SimSiamTransform(args.crop_size, train)
    else:
        train = False
        transform = SimSiamTransform(args.crop_size, train)

    dataset = torchvision.datasets.CIFAR10(root="CIFAR10_Dataset", train=train, transform=transform, download=download)

    if debug_subset_size is not None:
        dataset = torch.utils.data.Subset(dataset, range(0, debug_subset_size)) # take only one batch
        dataset.classes = dataset.dataset.classes
        dataset.targets = dataset.dataset.targets
    return dataset