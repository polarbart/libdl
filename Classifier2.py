import numpy as np
from pylibdl.data import Dataset, DataLoader
from utils import DistractedDriver, MyResNet
from pylibdl.optim import Adam
from pylibdl import cross_entropy_with_logits
import time
np.random.seed(0)


def validate():
    accs = []
    for imgs, labels in val_loader:
        pred = model(imgs).numpy()
        accs.append((pred.argmax(0) == labels.numpy().argmax(0)).mean())
    return np.mean(accs)


path = '/home/superbabes/Downloads/ddriver'
batch_size = 64
lr = 1e-4

train_data = DistractedDriver(path, val=False)
train_loader = DataLoader(train_data, batch_size, shuffle=True, drop_last=True)
val_data = DistractedDriver(path, val=True)
val_loader = DataLoader(val_data, 512, shuffle=False, drop_last=False)

model = MyResNet()
optim = Adam(model.parameter(), lr)

last = time.time()

for i, (imgs, labels) in enumerate(train_loader):
    pred = model(imgs)
    loss = cross_entropy_with_logits(pred, labels)
    loss.backward()
    optim.step()
    optim.zero_grad()
    print(i, loss.numpy())
    if i == 150:
        break
    if i % 25 == 0 and i > 0:
        print(i, validate(), time.time() - last)
model.save('model2')
print(validate())
print(time.time() - last)
