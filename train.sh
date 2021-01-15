##!/bin/sh

python train.py --cifar10 --batch-size 512 --lr 1e-2 --SimSiam --crop-size 32 --prediction --projection --arch resnet18 ## resnet18

python train.py --cifar10 --batch-size 512 --lr 1e-2 --SimSiam --crop-size 32 --prediction --projection --arch vgg19_bn ## vgg19_bn
#python linear_eval.py --save-dir D:/exp_results/0114-223028-vgg19_bn --arch vgg19_bn --crop-size 32

python train.py --cifar10 --batch-size 512 --lr 1e-2 --SimSiam --crop-size 32 --prediction --projection --arch vgg19_bn ## vgg19
