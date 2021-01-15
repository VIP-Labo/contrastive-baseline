import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision.models as models

class LinearEvalModel(nn.Module):
    def __init__(self, arch='vgg19', dim=512, num_classes=10):
        super().__init__()

        if arch == 'vgg19':
            self.features = models.vgg19().features
        if arch == 'vgg19_bn':
            self.features = models.vgg19_bn().features
        elif arch == 'resnet18':
            resnet18 = models.resnet18(pretrained=False)
            self.features = nn.Sequential(*list(resnet18.children())[:-1])

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Linear(dim, num_classes)

    def weight_init(self, weight_path, device):
        state_dict = torch.load(os.path.join(weight_path, 'best_model.pth'), device)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'encoder' in k:
                k = k.replace('encoder.', '')
                new_state_dict[k] = v
        
        self.features.load_state_dict(new_state_dict)

        for m in self.features.parameters():
            m.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.squeeze()
        out = self.fc(x)

        return out
