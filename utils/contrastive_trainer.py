import os
import sys
import time
import logging

import numpy as np

import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.datasets as datasets

from models.siamese_net import SiameseNetwork
from models.contrastive_loss import ContrastiveLoss
from utils.trainer import Trainer
from utils.helper import Save_Handle, AverageMeter, worker_init_fn
from utils.visualizer import ImageDisplayer, EmbeddingDisplayer
from datasets.spatial import SpatialDataset
from datasets.cifar10 import PosNegCifar10

class CoTrainer(Trainer):
    def setup(self):
        """initialize the datasets, model, loss and optimizer"""
        args = self.args
        self.vis = ImageDisplayer(args, self.save_dir)
        self.emb = EmbeddingDisplayer(args, self.save_dir)
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
        else:
            raise Exception("gpu is not available")

        if args.cifar10:
            # Download and create datasets
            or_train = datasets.CIFAR10(root="CIFAR10_Dataset", train=True, transform=None, download=True)
            or_val = datasets.CIFAR10(root="CIFAR10_Dataset", train=False, transform=None, download=True)

            # splits CIFAR10 into two streams
            # 20% of the images will be used as a stream for positives and negatives
            # the remaining images are used as anchor images

            self.datasets = {x: PosNegCifar10((or_train if x == 'train' else or_val),
                                              phase=x) for x in ['train', 'val']}
        else:
            self.datasets = {x: SpatialDataset(os.path.join(args.data_dir, x),
                                            args.crop_size,
                                            args.div_num,
                                            args.aug) for x in ['train', 'val']}

        self.dataloaders = {x: DataLoader(self.datasets[x],
                                        batch_size=args.batch_size,
                                        shuffle=(True if x == 'train' else False),
                                        num_workers=args.num_workers*self.device_count,
                                        pin_memory=(True if x == 'train' else False),
                                        worker_init_fn=worker_init_fn) for x in ['train', 'val']}

        #print("creating model '{}'".format(args.arch))
        self.model = SiameseNetwork(models.__dict__[args.arch], pretrained=args.imagenet, simple_model=args.simple_model)
        self.model.to(self.device)

        self.criterion = ContrastiveLoss(args.margin)
        self.criterion.to(self.device)

        self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum)

        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[40, 80, 120, 160, 200, 250], gamma=0.1)

        self.start_epoch = 0
        self.best_loss = np.inf
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, self.device))

        self.save_list = Save_Handle(max_num=args.max_model_num)

    def train(self):
        """training process"""
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            self.epoch = epoch
            self.train_epoch(epoch)
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                self.val_epoch(epoch)

    def train_epoch(self, epoch):
        epoch_loss = AverageMeter()
        epoch_start = time.time()
        self.model.train()  # Set model to training mode

        for step, (input1, input2, target, label) in enumerate(self.dataloaders['train']):
            input1 = input1.to(self.device)
            input2 = input2.to(self.device)
            target = target.to(self.device)

            with torch.set_grad_enabled(True):
                output1, output2 = self.model(input1, input2)
                loss = self.criterion(output1, output2, target)
                epoch_loss.update(loss.item(), input1.size(0))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # visualize
            if step == 0:
                self.vis(epoch, 'train', input1, input2, target)
                self.emb(output1, label, epoch, 'train')

        logging.info('Epoch {} Train, Loss: {:.5f}, Cost {:.1f} sec'
                     .format(self.epoch, epoch_loss.get_avg(), time.time()-epoch_start))

        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
        torch.save({
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': model_state_dic
        }, save_path)
        self.save_list.append(save_path)  # control the number of saved models

    def val_epoch(self, epoch):
        epoch_start = time.time()
        self.model.eval()  # Set model to evaluate mode
        epoch_loss = AverageMeter()

        for step, (input1, input2, target, label) in enumerate(self.dataloaders['val']):
            input1 = input1.to(self.device)
            input2 = input2.to(self.device)
            target = target.to(self.device)
            with torch.set_grad_enabled(False):
                output1, output2 = self.model(input1, input2)
                loss = self.criterion(output1, output2, target)
                epoch_loss.update(loss.item(), input1.size(0))

            # visualize
            if step == 0:
                self.vis(epoch, 'val', input1, input2, target)
                self.emb(output1, label, epoch, 'val')

        logging.info('Epoch {} Val, Loss: {:.5f}, Cost {:.1f} sec'
                     .format(self.epoch, epoch_loss.get_avg(), time.time()-epoch_start))

        model_state_dic = self.model.state_dict()
        if self.best_loss > epoch_loss.get_avg():
            self.best_loss = epoch_loss.get_avg()
            logging.info("save min loss {:.2f} model epoch {}".format(self.best_loss, self.epoch))
            torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model.pth'))