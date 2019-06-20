import numpy as np
import libdl as l
import torch, torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader


# A small helperfunction that instanciates my own tensors from a numpy array
def from_numpy(a, requiresGrad=False):
    a = np.array(a)
    if len(a.shape) == 1:
        return l.Tensor1(np.asfortranarray(a).astype(np.float32), requiresGrad)
    elif len(a.shape) == 2:
        return l.Tensor2(np.asfortranarray(a).astype(np.float32), requiresGrad)
    elif len(a.shape) == 3:
        return l.Tensor3(np.asfortranarray(a).astype(np.float32), requiresGrad)
    elif len(a.shape) == 4:
        return l.Tensor4(np.asfortranarray(a).astype(np.float32), requiresGrad)


def uniform(shape):
    fan_in = shape[-2]


lr = .1
lr_decay = .99
filter_size = 5
hidden_units = [16, 32, 32]
log_every = 500
epochs = 10
batch_size = 64


train_loader = DataLoader(torchvision.datasets.MNIST('dataset/', train=True, download=True,
                                                     transform=torchvision.transforms.Compose([
                                                         torchvision.transforms.ToTensor(),
                                                         torchvision.transforms.Normalize(
                                                             (0.1307,), (0.3081,))
                                                     ])),
                          batch_size=batch_size, shuffle=False)

test_loader = DataLoader(
    torchvision.datasets.MNIST('dataset/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=False)


F1 = from_numpy(np.random.normal(0, 1 / np.sqrt(filter_size * filter_size), (filter_size, filter_size, 1, hidden_units[0])), True)
B1 = from_numpy(np.zeros(hidden_units[0]), True)
F2 = from_numpy(np.random.normal(0, 1 / np.sqrt(hidden_units[0] * filter_size * filter_size), (filter_size, filter_size, hidden_units[0], hidden_units[1])), True)
B2 = from_numpy(np.zeros(hidden_units[1]), True)

W3 = from_numpy(np.random.normal(0, 1 / np.sqrt(25 * hidden_units[1]), (hidden_units[2], 25 * hidden_units[1])), True)
B3 = from_numpy(np.zeros(hidden_units[2]), True)
W4 = from_numpy(np.random.normal(0, 1 / np.sqrt(hidden_units[2]), (10, hidden_units[2])), True)
B4 = from_numpy(np.zeros(10), True)
parameters = [F1, B1, F2, B2, W3, B3, W4, B4]

TF1 = torch.tensor(F1.numpy().transpose().copy(), requires_grad=True)
TB1 = torch.tensor(B1.numpy().copy(), requires_grad=True)
TF2 = torch.tensor(F2.numpy().transpose().copy(), requires_grad=True)
TB2 = torch.tensor(B2.numpy().copy(), requires_grad=True)

TW3 = torch.tensor(W3.numpy().transpose().copy(), requires_grad=True)
TB3 = torch.tensor(B3.numpy().copy(), requires_grad=True)
TW4 = torch.tensor(W4.numpy().transpose().copy(), requires_grad=True)
TB4 = torch.tensor(B4.numpy().copy(), requires_grad=True)
tparameters = [TF1, TB1, TF2, TB2, TW3, TB3, TW4, TB4]

x = from_numpy(np.random.rand(20, 20, 1, 1))
tx = torch.tensor(x.numpy().transpose().copy())


def tforward(x):
    h1 = F.leaky_relu(F.max_pool2d(F.conv2d(x, TF1, TB1, padding=2), 2, 2))
    h2 = F.leaky_relu(F.max_pool2d(F.conv2d(h1, TF2, TB2, padding=2), 2, 2))
    h2 = h2.view(-1, 25 * hidden_units[1])
    h3 = F.leaky_relu(torch.matmul(h2, TW3) + TB3)
    return torch.matmul(h3, TW4) + TB4


def forward(x):
    h1 = l.leakyRelu(l.maxpool2d(l.conv2d(x, F1, B1, int(filter_size / 2)), 2))
    h2 = l.leakyRelu(l.maxpool2d(l.conv2d(h1, F2, B2, int(filter_size / 2)), 2))
    h3 = l.leakyRelu(l.matmul(W3, l.reshape(h2, (25 * hidden_units[1], -1))) + B3)
    return l.matmul(W4, h3) + B4


yp = forward(x)
loss = l.crossEntropyWithLogits(yp, from_numpy(np.eye(10)[:1]))
typ = tforward(tx)
tloss = F.cross_entropy(typ, torch.tensor([0]))

opt = torch.optim.SGD(tparameters, lr=lr, momentum=.5)

losses = []
for i, (x, y) in enumerate(train_loader):
    x = x[:, :, 4:24, 4:24]
    yp = tforward(x)
    loss = F.cross_entropy(yp, y)
    loss.backward()
    opt.step()
    opt.zero_grad()
    losses.append(loss.detach().numpy())
    if i == 100:
        break
print(np.mean(losses))
