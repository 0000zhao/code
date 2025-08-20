import torch
import torch.nn as nn
import torch.nn.functional as F

class GammaMapGenerator(nn.Module):
    def __init__(self, input_channels=3):
        super(GammaMapGenerator, self).__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.net2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.net3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.net4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.net5 = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        #h, w = x.shape[2], x.shape[3]

        x1 = self.net1(x)
        #x1 = F.softmax( self.advavg(x1), dim=1 ) * x1

        x2 = self.net2(x1)
        x2     = F.softmax( self.advavg(x2), dim=1 ) * x2

        x3 = self.net3(x2)
        #x3    = F.softmax( self.advavg(x3), dim=1 ) * x3

        x4 = self.net4(x3)
        x4 = F.softmax( self.advavg(x4), dim=1 ) * x4

        #x5= self.net5(x4)
        gamma_map = self.net5(x4)

        gamma_map = torch.where(gamma_map >0, gamma_map * 10, gamma_map)
        return gamma_map

class GammaCorrectionNetwork(nn.Module):
    def __init__(self):
        super(GammaCorrectionNetwork, self).__init__()
        self.gamma_generator = GammaMapGenerator()

    def forward(self, x):
        gamma_map = self.gamma_generator(x)
        enhance_image = torch.pow(x, 1 + gamma_map * torch.pow(x, 0.13))
        return enhance_image, gamma_map