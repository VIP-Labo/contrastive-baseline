##!/bin/sh

python train.py --cifar10 --batch-size 256 --lr 1e-2 --SimSiam --crop-size 32 --prediction --projection --arch resnet18 ## resnet18

python train.py --cifar10 --batch-size 256 --lr 1e-2 --SimSiam --crop-size 32 --prediction --projection --arch vgg19_bn ## vgg19_bn

python train.py --cifar10 --batch-size 256 --lr 1e-2 --SimSiam --crop-size 32 --prediction --projection --arch vgg19_bn ## vgg19
