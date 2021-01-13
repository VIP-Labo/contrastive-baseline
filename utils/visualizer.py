import os
import numpy as np
from PIL import Image

import torch

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

### torch テンソル(バッチ)を受け取って、args.div_numに応じて、描画する

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def invnorm(img, N):
    img = img[N,:,:,:].to('cpu').detach().numpy().copy()
    img = img.transpose(1,2,0)
    img = img*std+mean
    return img

class ImageDisplayer:
    def __init__(self, args, save_fir):
        # N is number of batch to display
        self.args = args
        self.save_dir = save_fir
        self.N = args.visual_num

    @torch.no_grad()
    def __call__(self, epoch, prefix, img1, img2, target):
        imgs1 = []
        imgs2 = []
        targets = []
        for n in range(self.N):
            imgs1.append(invnorm(img1,n))
            imgs2.append(invnorm(img2,n))
            targets.append(target[n].item())

        self.display_images(epoch, prefix, imgs1, imgs2, targets)

    def display_images(self, epoch, prefix, images1: [Image], images2: [Image], targets, 
                       columns=2, width=8, height=8, label_wrap_length=50, label_font_size=8):

        if not (images1 and images2):
            print("No images to display.")
            return 

        height = max(height, int(len(images1)/columns) * height)
        plt.figure(figsize=(width, height))
        i = 1
        for (im1, im2, tar) in zip(images1, images2, targets):
            im1 = Image.fromarray(np.uint8(im1*255))
            im2 = Image.fromarray(np.uint8(im2*255))

            plt.subplot(self.N, 2, i)
            plt.title(tar, fontsize=20) 
            plt.imshow(im1)
            i += 1
            plt.subplot(self.N, 2, i)
            plt.title(tar, fontsize=20) 
            plt.imshow(im2)
            i += 1
        
        plt.tight_layout()
        output_img_name = 'imgs_{}_{}.png'.format(prefix, epoch)
        plt.savefig(os.path.join(self.save_dir, 'images', output_img_name))
        plt.close()

class EmbeddingDisplayer:
    def __init__(self, args, save_fir):

        self.args = args
        self.save_dir = save_fir
        self.cifar10_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

    @torch.no_grad()
    def __call__(self, embeddings, targets, epoch, prefix, xlim=None, ylim=None):
        embeddings = embeddings.to('cpu').detach().numpy().copy()
        targets = targets.to('cpu').detach().numpy().copy()
        plt.figure(figsize=(10,10))
        for i in range(10):
            inds = np.where(targets==i)[0]
            plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=self.colors[i])
        if xlim:
            plt.xlim(xlim[0], xlim[1])
        if ylim:
            plt.ylim(ylim[0], ylim[1])
        plt.legend(self.cifar10_classes)
        output_img_name = 'emb_{}_{}.png'.format(prefix, epoch)
        plt.savefig(os.path.join(self.save_dir, 'images', output_img_name))
        plt.close()

class GraphPloter:
    def __init__(self, args, save_fir, prefix):
        self.args = args
        self.save_dir = save_fir
        self.epochs = []
        self.losses = []

    def __call__(self, epoch, loss):
        self.epochs.append(epoch)
        self.losses.append(loss)
        output_img_name = '{}_loss.svg'.format(prefix)

        plt.plot(self.epochs, self.losses)
        plt.title('Loss')
        plt.savefig(os.path.join(self.save_dir, 'images', output_img_name))
        plt.close()

