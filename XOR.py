import numpy as np
import libdl as l


def from_numpy(a, requiresGrad=False):
    a = np.array(a)
    if len(a.shape) == 1:
        return l.Tensor1(np.asfortranarray(a).astype(np.float32), requiresGrad)
    elif len(a.shape) == 2:
        return l.Tensor2(np.asfortranarray(a).astype(np.float32), requiresGrad)


np.random.seed(3)

x = from_numpy([[0, 0], [0, 1], [1, 0], [1, 1]])
y = from_numpy([[0], [1], [1], [0]])


lr = .005
hidden_layer = 2

W1 = from_numpy(np.random.normal(0, 1/2, (2, hidden_layer)), True)
B1 = from_numpy(np.zeros(hidden_layer), True)
W2 = from_numpy(np.random.normal(0, 1/hidden_layer, (hidden_layer, 1)), True)
B2 = from_numpy([0], True)
parameters = [W1, B1, W2, B2]


def forward(x):
    h1 = l.sigmoid(l.add(l.matmul(x, W1), B1))
    return l.sigmoid(l.add(l.matmul(h1, W2), B2))


for epoch in range(1000):
    yp = forward(x)
    print(yp.data)
    loss = l.sum(l.sum(l.pow(l.sub(y, yp), 2), 0), 0)
    print(loss.data)
    loss.backward(1)
    for p in parameters:
        p.applyGradient(lr)
        p.zeroGrad()
    lr *= .95

