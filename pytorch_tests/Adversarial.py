import torch
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms
from DistracedDriver import DistractedDriver
from torch.nn import CrossEntropyLoss
import numpy as np
import pickle
from MyResnet import ResNet

path = '/home/superbabes/Downloads/ddriver'
lr = 1

transforms = transforms.Compose([transforms.ToPILImage(), transforms.Resize((128, 128)), transforms.ToTensor()])

val_data = DistractedDriver(path, val=True, transforms=transforms)

loss_fun = CrossEntropyLoss()

with open('model', 'rb') as f:
    model = pickle.load(f)

model.cpu()
for p in model.parameters():
    p.requires_grad = False

imgs = [val_data[i*128][0] for i in range(10)]
imgs_tensor = torch.tensor(np.stack(imgs, axis=0))
eps = torch.zeros_like(imgs_tensor, requires_grad=True)

target = torch.tensor([0] * imgs_tensor.shape[0])
print(target.numpy())
for i in range(50):
    pred = model(imgs_tensor + eps)
    ce = loss_fun(pred, target)
    loss = ce + torch.mean(eps**2)
    loss.backward()
    eps.data -= lr * eps.grad
    eps.grad.zero_()
    model.zero_grad()
    print(ce.detach().numpy(), loss.detach().numpy(), pred.detach().numpy().argmax(1))

