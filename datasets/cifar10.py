from typing import Callable, Optional
import random

from PIL import Image
import numpy as np

import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10

np.random.seed(765)
random.seed(765)

train_transform = transforms.Compose([transforms.Resize(64),
                        transforms.RandomResizedCrop(scale=(0.16, 1), ratio=(0.75, 1.33), size=64),
                        transforms.RandomHorizontalFlip(0.5),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

val_transform = transforms.Compose([transforms.Resize(64),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

class SupervisedPosNegCifar10(torch.utils.data.Dataset):
    def __init__(self, dataset, phase):
        # split by some thresholds here 80% anchors, 20% for posnegs
        lengths = [int(len(dataset)*0.8), int(len(dataset)*0.2)]
        self.anchors, self.posnegs = torch.utils.data.random_split(dataset, lengths)
        
        if phase == 'train':
            self.anchor_transform = train_transform
            self.posneg_transform = train_transform
        else:
            self.anchor_transform = val_transform
            self.posneg_transform = val_transform

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
            self.anchor_transform = train_transform
            self.posneg_transform = train_transform
        else:
            self.anchor_transform = val_transform
            self.posneg_transform = val_transform

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