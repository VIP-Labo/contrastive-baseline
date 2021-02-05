#!/bin/sh
#python train.py --data-dir /mnt/hdd02/shibuya_scramble --save-dir /mnt/hdd02/contrastive-learn/shibuya --SimSiam --arch vgg19_bn --projection --prediction --mlp-bn --aug --batch-size 64

python train.py --data-dir /mnt/hdd02/shibuya_scramble --save-dir /mnt/hdd02/contrastive-learn/shibuya --SimSiam --arch vgg19_bn --projection --prediction --mlp-bn --imagenet --aug --batch-size 64

python train.py --data-dir /mnt/hdd02/shibuya_scramble --save-dir /mnt/hdd02/contrastive-learn/shibuya --SimSiam --arch vgg19 --projection --prediction --mlp-bn --imagenet --aug --batch-size 64