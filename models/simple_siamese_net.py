import torch
import torch.nn as nn

class projection_MLP(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=512, out_dim=512): # bottleneck structure
        super().__init__()
        
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
    def __init__(self, in_dim=512, hidden_dim=256, out_dim=512): # bottleneck structure
        super().__init__()
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
    def __init__(self, model, pattern_feature = 'conv-512x1x1', projection=False, prediction=False):
        super(SiameseNetwork, self).__init__()
        self.projection = projection
        self.prediction = prediction

        if pattern_feature == 'conv-512x1x1':
            features = model().features
            max_pool = nn.AdaptiveAvgPool2d((1,1))
            self.encoder = nn.Sequential(features, max_pool)

            if projection:
                self.projector = projection_MLP(in_dim=512, hidden_dim=512, out_dim=512)

            if prediction:
                self.predictor = prediction_MLP(in_dim=512, out_dim=512)

        #elif pattern_feature == 'conv-512x7x7': ## Not Yet
        #    self.encoder = model().features

        elif pattern_feature == 'fc-4096':
            features = model()
            self.encoder = nn.Sequential(*[self.encoder.classifier[0]])

            if projection:
                self.projector = projection_MLP(in_dim=4096, hidden_dim=4096, out_dim=4096)
            
            if prediction:
                self.predictor = prediction_MLP(in_dim=4096, out_dim=4096)


    def forward(self, input1, input2):
        if self.prediction:
            f, h = self.encoder, self.predictor
            z1, z2 = f(input1), f(input2)

            if self.projection:
                z1, z2 = self.projection(input1), self.projection(input2)

            p1, p2 = h(z1), h(z2)

        else:
            f = self.encoder
            z1, z2 = f(input1), f(input2)

            if self.projection:
                z1, z2 = self.projection(input1), self.projection(input2)

            p1, p2 = None, None

        return (z1, z2), (p1, p2)

