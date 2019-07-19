import numpy as np
import libdl
from pylibdl.tensor import tensor


class Adam:

    def __init__(self, parameter, lr=1e-3, b1=.9, b2=.999, eps=1e-8):
        self.parameter = [(p, tensor(np.zeros(p.shape)), tensor(np.zeros(p.shape))) for p in parameter if p.requires_grad]
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps

    def step(self):
        for p, m, v in self.parameter:
            libdl.apply_adam(p, m, v, self.lr, self.b1, self.b2, self.eps)

    def zero_grad(self):
        for p, _, _ in self.parameter:
            p.zero_grad()
