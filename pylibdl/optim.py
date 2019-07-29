import pylibdl as libdl
from pylibdl import zeros, Tensor
from typing import List


class Adam:
    """
    Adam optimizer
    """
    def __init__(self, parameter: List[Tensor], lr: float = 1e-3, b1: float = .9, b2: float = .999, eps: float = 1e-8):
        self.parameter = [(p, zeros(p.shape, False), zeros(p.shape, False)) for p in parameter if p.requires_grad]
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
