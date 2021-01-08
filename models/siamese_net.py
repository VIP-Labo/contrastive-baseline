import torch
import torch.nn as nn

class SiameseNetwork(nn.Module):
    def __init__(self, model):
        super(SiameseNetwork, self).__init__()
        self.encoder = model(num_classes=2)

    def forward_once(self, x):
        output = self.encoder(x)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2