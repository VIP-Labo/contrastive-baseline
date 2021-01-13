##!/bin/sh

python train.py --cifar10 --batch-size 64 --lr 1e-2 --SimSiam --crop-size 224 --prediction
python train.py --cifar10 --batch-size 64 --lr 1e-2 --SimSiam --crop-size 224

python train.py --cifar10 --batch-size 64 --lr 1e-2 --SimSiam --crop-size 224 --pattern-feature fc-4096 --prediction
python train.py --cifar10 --batch-size 64 --lr 1e-2 --SimSiam --crop-size 224 --pattern-feature fc-4096
