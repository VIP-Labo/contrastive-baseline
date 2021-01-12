import torch
import torch.nn as nn

class SiameseNetwork(nn.Module):
    def __init__(self, model, pretrained=False, simple_model=False):
        super(SiameseNetwork, self).__init__()
        self.simple_model = simple_model
        if simple_model:
            self.features = nn.Sequential(nn.Conv2d(3, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(64, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

            self.classifier = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
                                     nn.PReLU(),
                                     nn.Linear(256, 256),
                                     nn.PReLU(),
                                     nn.Linear(256, 2))

        else:
            if pretrained:
                self.encoder = model(pretrained=True)
                self.encoder.classifier = nn.Sequential(*[self.encoder.classifier[i] for i in range(6)])
                self.encoder.classifier.add_module('out', nn.Linear(4096, 2))
            else:
                self.encoder = model(num_classes=2)

    def forward_once(self, x):
        if self.simple_model:
            output = self.features(x)
            output = output.view(output.size()[0], -1)
            output = self.classifier(output)

        else:
            output = self.encoder(x)
        
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2