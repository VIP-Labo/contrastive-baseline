python train.py --cifar10 --batch-size 512 --lr 1e-2 --SimSiam --crop-size 32 --prediction --projection --arch resnet18
python linear_eval.py --save-dir D:/exp_results/resnet18 --arch resnet18 --crop-size 32

python train.py --cifar10 --batch-size 512 --lr 1e-2 --SimSiam --crop-size 32 --prediction --projection --arch vgg19_bn
python linear_eval.py --save-dir D:/exp_results/vgg19_bn --arch vgg19_bn --crop-size 32

python train.py --cifar10 --batch-size 512 --lr 1e-2 --SimSiam --crop-size 32 --prediction --projection --arch vgg19
python linear_eval.py --save-dir D:/exp_results/vgg19 --arch vgg19 --crop-size 32