import numpy as np
import pickle
import os
import warnings
from skimage.io import imread
from skimage.transform import resize
from pylibdl.data import Dataset, DataLoader
import pylibdl as libdl
from pylibdl.modules import *
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pylibdl.data import DataLoader
from pylibdl.optim import Adam
from utils import DistractedDriver, validate
from pylibdl import cross_entropy_with_logits, mean
from pylibdl.tensor import tensor
import time
np.random.seed(0)

class MyResNet(Module):

    def __init__(self):
        super().__init__()

        self.initial = Sequential(
            Conv2D(3, 64, 5, stride=2, bias=False),  # 64x64
            BatchNorm2d(64),
            MaxPool2d(2),  # 32x32
            LeakyReLU()
        )
        #self.bn = BatchNorm2d(3)

        self.res1 = Sequential(
            Conv2D(64, 64, 3, stride=1, bias=False),
            BatchNorm2d(64),
            LeakyReLU(),
            Conv2D(64, 64, 3, stride=1, bias=False),  # 16x16
            BatchNorm2d(64)
        )
        self.adapt1 = Sequential(
            Conv2D(64, 64, 1, stride=1, bias=False),
            BatchNorm2d(64)
        )

        self.res2 = Sequential(
            Conv2D(64, 128, 3, stride=1, bias=False),
            BatchNorm2d(128),
            LeakyReLU(),
            Conv2D(128, 128, 3, stride=1, bias=False),
            BatchNorm2d(128)
        )
        self.adapt2 = Sequential(
            Conv2D(64, 128, 1, stride=1, bias=False),
            BatchNorm2d(128)
        )

        self.l2 = Linear(32*32*128, 10)

    def forward(self, x):
        x = self.initial(x)
        #print(np.allclose(x.numpy(), self.bn(x).numpy()))
        #x = self.bn(x)
        x = libdl.leaky_relu(self.res1(x) + self.adapt1(x)) # libdl.leaky_relu
        x = libdl.leaky_relu(self.res2(x) + self.adapt2(x))
        x = libdl.reshape(x, (32*32*128, -1))
        return self.l2(x)

path = '/home/superbabes/Downloads/ddriver'
val_data = DistractedDriver(path, val=True)
#model = MyResNet.load('model')
model = MyResNet()
model.eval()

imgs_tensor = tensor(np.stack([val_data[i*128][0] for i in range(6, 7)], axis=-1), requires_grad=True)
eps = tensor(np.zeros(imgs_tensor.shape), requires_grad=True)
target = tensor(np.eye(10)[[0] * imgs_tensor.shape[-1]].transpose())
#optim = Adam([eps], .1)
for i in range(500):
    pred = model(imgs_tensor + eps)
    ce = cross_entropy_with_logits(pred, target)
    loss = ce + mean(eps**2)
    loss.backward()
    #optim.step()
    #optim.zero_grad()
    eps.apply_gradient(5)
    eps.zero_grad()
    model.zero_grad()
    print(ce.numpy(), loss.numpy(), pred.numpy().argmax(0))
