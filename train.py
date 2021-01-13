from utils.contrastive_trainer import CoTrainer
from utils.simsiam_trainer import SimSiamTrainer
import argparse
import os
import math
import torch
args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train ')
    parser.add_argument('--data-dir', default='/mnt/hdd02/process-ucf',
                        help='training data directory')
    parser.add_argument('--save-dir', default='/mnt/hdd02/contrastive-learn',
                        help='directory to save models.')
    parser.add_argument('--cifar10', action='store_true',
                        help='use cifar10 dataset')

    parser.add_argument('--SimSiam', action='store_true',
                        help='try Simple Siamese Net')                 

    parser.add_argument('--arch', type=str, default='vgg19',
                        help='the model architecture')
    parser.add_argument('--pattern-feature', type=str, default='conv-512x1x1',
                        help='the feature to contrast [conv-512x1x1, fc-4096]')
    parser.add_argument('--prediction', action='store_true',
                        help='use MLP prediction')

    parser.add_argument('--lr', type=float, default=1e-5,
                        help='the initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='the weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='the momentum')

    parser.add_argument('--div-num', type=int, default=3,
                    help='one side`s number of pathes')
    parser.add_argument('--aug', action='store_true',
                        help='the weight decay')
    parser.add_argument('--margin', type=float, default=1.0,
                        help='the margin of loss function')                         

    parser.add_argument('--resume', default='',
                        help='the path of resume training model')
    parser.add_argument('--max-model-num', type=int, default=1,
                        help='max models num to save ')
    parser.add_argument('--max-epoch', type=int, default=300,
                        help='max training epoch')
    parser.add_argument('--val-epoch', type=int, default=10,
                        help='the num of steps to log training information')
    parser.add_argument('--val-start', type=int, default=0,
                        help='the epoch start to val')

    parser.add_argument('--batch-size', type=int, default=8,
                        help='train batch size')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='the num of training process')

    parser.add_argument('--crop-size', type=int, default=224,
                        help='the crop size of the train image')

    parser.add_argument('--visual-num', type=int, default=4,
                        help='the number of visualize images')                       

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip('-')  # set vis gpu
    if args.SimSiam:
        trainer = SimSiamTrainer(args)
    else:
        trainer = CoTrainer(args)
    trainer.setup()
    trainer.train()
