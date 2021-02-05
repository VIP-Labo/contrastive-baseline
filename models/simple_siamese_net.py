import torch
import torch.nn as nn
import torchvision.models as models

class projection_MLP(nn.Module):
    def __init__(self, bn=True, in_dim=512, hidden_dim=512, out_dim=512): # bottleneck structure
        super().__init__()
    
        if bn:
            self.layers = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim)
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim)
            )

    def forward(self, x):
        if x.dim() != 2:
            x = x.squeeze()
        x = self.layers(x)
        return x 

class prediction_MLP(nn.Module):
    def __init__(self, bn=True, in_dim=512, hidden_dim=256, out_dim=512): # bottleneck structure
        super().__init__()

        if bn:
            self.layer1 = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True)
            )
        else:
            self.layer1 = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True)
            )

        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        if x.dim() != 2:
            x = x.squeeze()
        x = self.layer1(x)
        x = self.layer2(x)
        return x 

class SiameseNetwork(nn.Module):
    def __init__(self, args):
        super(SiameseNetwork, self).__init__()
        self.projection = args.projection
        self.prediction = args.prediction
        m = models.__dict__[args.arch](args.imagenet)

        if args.pattern_feature == 'conv-512x1x1':
            if args.arch == 'resnet18':
                self.encoder = nn.Sequential(*list(m.children())[:-1])
                self.bn = args.mlp_bn

            if args.arch == 'vgg19_bn':
                features = m.features
                max_pool = nn.AdaptiveAvgPool2d((1,1))
                self.encoder = nn.Sequential(features, max_pool)
                self.bn = args.mlp_bn

            if args.arch == 'vgg19':
                features = m.features
                max_pool = nn.AdaptiveAvgPool2d((1,1))
                self.encoder = nn.Sequential(features, max_pool)
                self.bn = args.mlp_bn

            if self.projection:
                self.projector = projection_MLP(bn=self.bn, in_dim=512, hidden_dim=512, out_dim=512)

            if self.prediction:
                self.predictor = prediction_MLP(bn=self.bn, in_dim=512, out_dim=512)

        """
        elif args.pattern_feature == 'fc-4096':
            features = model()
            self.encoder = nn.Sequential(*[self.encoder.classifier[0]])

            if projection:
                self.projector = projection_MLP(in_dim=4096, hidden_dim=4096, out_dim=4096)
            
            if prediction:
                self.predictor = prediction_MLP(in_dim=4096, out_dim=4096)
        """

    def forward(self, input1, input2):
        if self.prediction:
            f, h = self.encoder, self.predictor
            z1, z2 = f(input1), f(input2)

            if self.projection:
                z1, z2 = self.projector(z1), self.projector(z2)

            p1, p2 = h(z1), h(z2)

        else:
            f = self.encoder
            z1, z2 = f(input1), f(input2)

            if self.projection:
                z1, z2 = self.projector(z1), self.projector(z2)

            p1, p2 = None, None

        return (z1, z2), (p1, p2)

