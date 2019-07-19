import numpy as np
from pylibdl import _tensor_types, tensor
import libdl
import pickle


class Module:

    def __init__(self):
        self.tensors = []
        self.modules = []
        self.is_train = True

    def train(self, is_train=True):
        self.is_train = is_train
        for m in self.modules:
            m.train(is_train)

    def eval(self):
        self.train(False)

    def zero_grad(self):
        for t in self.tensors + self.modules:
            t.zero_grad()

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __setattr__(self, key, value):
        self._register(value)
        super().__setattr__(key, value)

    def _register(self, v):
        if type(v) in _tensor_types:
            self.tensors.append(v)
        elif issubclass(type(v), Module):
            self.modules.append(v)

    def parameter(self):
        return [t for m in self.modules for t in m.parameter()] + self.tensors

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)


class Linear(Module):

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.w = tensor(np.random.normal(0, 1 / np.sqrt(in_features), size=(in_features, out_features)), requires_grad=True)
        self.b = tensor(np.zeros(out_features), requires_grad=True) if bias else None

    def forward(self, x):
        return libdl.linear(self.w, x, self.b)


class Conv2D(Module):

    def __init__(self, in_channels, out_channels, filter_size, padding=None, stride=1, bias=True):
        super().__init__()
        if padding is None:
            padding = filter_size // 2
        self.padding = padding
        self.stride = stride
        self.filter = tensor(np.random.normal(0, 1 / np.sqrt(in_channels * filter_size * filter_size), size=(in_channels, filter_size, filter_size, out_channels)), requires_grad=True)
        self.bias = tensor(np.zeros(out_channels), requires_grad=True) if bias else None

    def forward(self, x):
        return libdl.conv_2d(x, self.filter, self.bias, self.padding, self.stride)


class BatchNorm2d(Module):

    def __init__(self, num_features, momentum=.9, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.gamma = tensor(np.ones(num_features), requires_grad=True)
        self.beta = tensor(np.zeros(num_features), requires_grad=True)
        self.running_mean = tensor(np.zeros(num_features), requires_grad=False)
        self.running_var = tensor(np.ones(num_features), requires_grad=False)

    def forward(self, x):
        return libdl.batch_norm_2d(x, self.gamma, self.beta, self.running_mean, self.running_var, self.momentum, self.eps, not self.is_train)


class ReLu(Module):

    def forward(self, x):
        return libdl.relu(x)


class LeakyReLU(Module):

    def __init__(self, negative_slope=.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        return libdl.leaky_relu(x, self.negative_slope)


class MaxPool2d(Module):

    def __init__(self, kernel_size_and_stride=2):
        super().__init__()
        self.kernel_size_and_stride = kernel_size_and_stride

    def forward(self, x):
        return libdl.maxpool_2d(x, self.kernel_size_and_stride)


class Sequential(Module):
    def __init__(self, *seq):
        super().__init__()
        self.seq = seq
        for m in self.seq:
            self._register(m)

    def forward(self, x):
        for m in self.seq:
            x = m(x)
        return x
