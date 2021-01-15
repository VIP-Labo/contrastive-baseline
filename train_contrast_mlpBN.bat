
python linear_eval.py --save-dir D:/exp_results_contrast_bn/resnet18 --arch resnet18 --crop-size 32 --lr 1e-1

python train.py --save-dir D:/exp_results_contrast_bn --cifar10 --batch-size 512 --lr 1e-2 --SimSiam --crop-size 32 --prediction --projection --arch vgg19 --mlp-bn
python linear_eval.py --save-dir D:/exp_results_contrast_bn/vgg19 --arch vgg19 --crop-size 32 --lr 1e-1

python train.py --save-dir D:/exp_results_contrast_bn --cifar10 --batch-size 512 --lr 1e-2 --SimSiam --crop-size 32 --prediction --projection --arch vgg19_bn
python linear_eval.py --save-dir D:/exp_results_contrast_bn/vgg19_bn --arch vgg19_bn --crop-size 32 --lr 1e-1