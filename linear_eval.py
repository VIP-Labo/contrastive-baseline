import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision.models as models

from datasets.cifar10 import get_simsiam_dataset
from models.create_linear_eval_model import LinearEvalModel
from utils.visualizer import AccLossGraphPloter

args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--save-dir', default='/mnt/hdd02/contrastive-learn/0113-193048 (vgg not BN not projector))',
                        help='model directory')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--arch', default='vgg19', help='model architecture')

    parser.add_argument('--max-epoch', default=100, help='train epoch')
    parser.add_argument('--crop-size', default=224, type=int, help='input size')
    parser.add_argument('--batch-size', default=512, type=int, help='input size')
    parser.add_argument('--lr', default=1e-2, help='learning rate')
    parser.add_argument('--momentum', default=0.9, help='momentum')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    plotter = AccLossGraphPloter(args.save_dir)

    datasets = {x: get_simsiam_dataset(args, x) for x in ['linear_train', 'linear_val']}

    dataloaders = {x: DataLoader(datasets[x],
                                batch_size=(args.batch_size),
                                shuffle=(True if x == 'linear_train' else False),
                                num_workers=8,
                                pin_memory=(True if x == 'linear_train' else False)) for x in ['linear_train', 'linear_val']}

    device = torch.device('cuda')

    model = LinearEvalModel(arch=args.arch)
    model.weight_init(args.save_dir, device) ## initialize & freeze

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[40, 60, 80], gamma=0.1)

    ## Training & Test Roop
    model.to(device)
    for epoch in range(args.max_epoch):
        model.train()
        losses, acc, step, total = 0., 0., 0., 0.
        for data, target in dataloaders['linear_train']:
            data, target = data.to(device), target.to(device)

            logits = model(data)

            optimizer.zero_grad()
            loss = criterion(logits, target)
            loss.backward()
            losses += loss.item()
            optimizer.step()
            scheduler.step()

            pred = F.softmax(logits, dim=-1).max(-1)[1]
            acc += pred.eq(target).sum().item()

            step += 1
            total += target.size(0)

        tr_loss = losses / step
        tr_acc = acc / total * 100.
        print('[Train Epoch: {0:2d}], loss: {1:.3f}, acc: {2:.3f}'.format(epoch, tr_loss, tr_acc))
        
        model.eval()
        losses, acc, step, total = 0., 0., 0., 0.
        with torch.no_grad():
            for data, target in dataloaders['linear_val']:
                data, target = data.to(device), target.to(device)

                logits = model(data)
                loss = criterion(logits, target)
                losses += loss.item()

                pred = F.softmax(logits, dim=-1).max(-1)[1]
                acc += pred.eq(target).sum().item()

                step += 1
                total += target.size(0)
            
            vl_loss = losses / step
            vl_acc = acc / total * 100.
            print('[Test Epoch: {0:2d}], loss: {1:.3f} acc: {2:.2f}'.format(epoch, vl_loss, vl_acc))

        plotter(epoch, tr_acc, vl_acc, tr_loss, vl_loss, args.arch)