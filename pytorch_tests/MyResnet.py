import torch
from torch.nn import *
from torch.nn.functional import leaky_relu


class ResNet(Module):

    def __init__(self):
        super().__init__()
        self.initial = Sequential(
            Conv2d(3, 64, 5, padding=2, stride=2, bias=False),  # 64x64
            BatchNorm2d(64),
            MaxPool2d(2),  # 32x32
            LeakyReLU()
        )

        self.res1 = Sequential(
            Conv2d(64, 64, 3, padding=1, stride=1, bias=False),
            BatchNorm2d(64),
            LeakyReLU(),
            Conv2d(64, 64, 3, padding=1, stride=2, bias=False),  # 16x16
            BatchNorm2d(64)
        )
        self.adapt1 = Sequential(
            Conv2d(64, 64, 1, padding=0, stride=2, bias=False),
            BatchNorm2d(64)
        )

        self.res2 = Sequential(
            Conv2d(64, 128, 3, padding=1, stride=1, bias=False),
            BatchNorm2d(128),
            LeakyReLU(),
            Conv2d(128, 128, 3, padding=1, stride=1, bias=False),
            BatchNorm2d(128)
        )
        self.adapt2 = Sequential(
            Conv2d(64, 128, 1, padding=0, stride=1, bias=False),
            BatchNorm2d(128)
        )

        self.l2 = Linear(16*16*128, 10)

    def forward(self, x):
        x = self.initial(x)
        x = leaky_relu(self.res1(x) + self.adapt1(x))
        x = leaky_relu(self.res2(x) + self.adapt2(x))
        x = x.view((-1, 16*16*128))
        return self.l2(x)
