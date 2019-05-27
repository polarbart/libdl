import numpy as np
import libdl as l
np.random.seed(0)


# A small helperfunction that instanciates my own tensors from a numpy array
def from_numpy(a, requiresGrad=False):
    a = np.array(a)
    if len(a.shape) == 1:
        return l.Tensor1(np.asfortranarray(a).astype(np.float32), requiresGrad)
    elif len(a.shape) == 2:
        return l.Tensor2(np.asfortranarray(a).astype(np.float32), requiresGrad)


# the dataset
x = from_numpy([[0, 0], [0, 1], [1, 0], [1, 1]])
y = from_numpy([[0], [1], [1], [0]])

# hyperparameter
lr = .001
lr_decay = .999
hidden_neurons = 2
log_every = 500
epochs = 10000

# weights and biases for the neural network
W1 = from_numpy(np.random.normal(0, 1 / 2, (2, hidden_neurons)), True)
B1 = from_numpy(np.zeros(hidden_neurons), True)
W2 = from_numpy(np.random.normal(0, 1 / hidden_neurons, (hidden_neurons, 1)), True)
B2 = from_numpy([0], True)
parameters = [W1, B1, W2, B2]


# forward pass
def forward(x):
    h1 = l.sigmoid(l.matmul(x, W1) + B1)
    return l.sigmoid(l.matmul(h1, W2) + B2)


# training
print("epoch |  0^0 |  0^1 |  1^0 |  1^1 | loss")
for epoch in range(epochs):

    yp = forward(x)
    loss = l.mean((y - yp)**2)  # mse
    loss.backward()

    for p in parameters:
        p.applyGradient(lr)
        p.zeroGrad()
    lr *= lr_decay

    if (epoch % log_every) == 0 or epoch == (epochs - 1):
        print("{:5d} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.6f}".format(epoch, *yp.numpy()[:, 0], loss.numpy()))