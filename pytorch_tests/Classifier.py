import torch
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms
from DistracedDriver import DistractedDriver
from torch.nn import CrossEntropyLoss
import numpy as np
import pickle
from MyResnet import ResNet


def validate():
    with torch.no_grad():
        model.eval()
        accs = []
        losses = []
        for imgs, labels in val_loader:
            pred = model(imgs.cuda()).cpu()
            losses.append(loss_fun(pred, labels))
            accs.append((pred.argmax(dim=1) == labels).float().mean())
        model.train()
        return np.mean(accs), np.mean(losses)

path = '/home/superbabes/Downloads/ddriver'
lr = 1e-3
batch_size = 64

transforms = transforms.Compose([transforms.ToPILImage(), transforms.Resize((128, 128)), transforms.ToTensor()])

train_set = DistractedDriver(path, val=False, transforms=transforms)
val_set = DistractedDriver(path, val=True, transforms=transforms)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=6)
val_loader = DataLoader(val_set, batch_size=512, shuffle=False, drop_last=False, num_workers=6)

loss_fun = CrossEntropyLoss()

model = ResNet()
model = model.cuda()

optim = torch.optim.Adam(model.parameters(), lr=lr)
#lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda x: 1. if x < 100 else .5 if x < 250 else .1)

for e in range(2):
    for i, (imgs, labels) in enumerate(train_loader):
        pred = model(imgs.cuda())
        loss = loss_fun(pred, labels.cuda())
        loss.backward()
        optim.step()
        optim.zero_grad()
        #lr_scheduler.step()
        if i > 100:
            break
print(validate())
with open('model', 'wb') as f:
    pickle.dump(model, f)
