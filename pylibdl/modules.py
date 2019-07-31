import numpy as np
import pylibdl as libdl
from pylibdl import _tensor_types, normal, zeros, ones, Tensor
import pickle
from typing import Any, Optional


class Module:
    '''
    A module represents a neural network with its weights, biases, forward pass, ...
    A module can in it self contain other modules
    '''
    def __init__(self):
        # the parameter of this network i.e. the weights, biases, ...
        self.tensors = []  
        # the modules this module contains
        self.modules = []
        # train mode, important for batch norm
        self.is_train = True

    # set train mode
    def train(self, is_train: bool = True):
        self.is_train = is_train
        for m in self.modules:
            m.train(is_train)

    def eval(self):
        self.train(False)
    
    # reset the gradient for all parameters
    def zero_grad(self):
        for t in self.tensors + self.modules:
            t.zero_grad()

    # the forward pass
    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __setattr__(self, key, value):
        self._register(value)
        super().__setattr__(key, value)

    def _register(self, v: Any):
        if type(v) in _tensor_types:
            self.tensors.append(v)
        elif issubclass(type(v), Module):
            self.modules.append(v)
    
    # return all paremeters (weights, biases, ...) this module contains
    def parameter(self):
        return [t for m in self.modules for t in m.parameter()] + self.tensors

    # save the module
    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    # load a saved module
    @staticmethod
    def load(path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)


class Linear(Module):

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.w = normal([in_features, out_features], 0, 1 / np.sqrt(in_features), True)
        self.b = zeros([out_features], True) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        return libdl.linear(self.w, x, self.b)


class Conv2d(Module):

    def __init__(self, in_channels: int, out_channels: int, filter_size: int, padding: Optional[int] = None, stride: int = 1, bias: bool = True):
        super().__init__()
        if padding is None:
            padding = filter_size // 2
        self.padding = padding
        self.stride = stride
        self.filter = normal([in_channels, filter_size, filter_size, out_channels], 0, 1 / np.sqrt((in_channels * filter_size * filter_size)), True)
        self.bias = zeros(out_channels, True) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        return libdl.conv_2d(x, self.filter, self.bias, self.padding, self.stride)


class BatchNorm2d(Module):

    def __init__(self, num_features: int, momentum: float = .9, eps: float = 1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.gamma = ones([num_features], True)
        self.beta = zeros([num_features], True)
        self.running_mean = zeros([num_features], False)
        self.running_var = ones([num_features], False)

    def forward(self, x: Tensor) -> Tensor:
        return libdl.batch_norm_2d(x, self.gamma, self.beta, self.running_mean, self.running_var, self.momentum, self.eps, not self.is_train)


class ReLu(Module):

    def forward(self, x: Tensor) -> Tensor:
        return libdl.relu(x)


class LeakyReLU(Module):

    def __init__(self, negative_slope: float = .01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: Tensor) -> Tensor:
        return libdl.leaky_relu(x, self.negative_slope)


class Sigmoid(Module):

    def forward(self, x: Tensor) -> Tensor:
        return libdl.sigmoid(x)


class MaxPool2d(Module):

    def __init__(self, kernel_size_and_stride: int = 2):
        super().__init__()
        self.kernel_size_and_stride = kernel_size_and_stride

    def forward(self, x: Tensor) -> Tensor:
        return libdl.maxpool_2d(x, self.kernel_size_and_stride)


class Sequential(Module):
    def __init__(self, *seq: Tensor):
        super().__init__()
        self.seq = seq
        for m in self.seq:
            self._register(m)

    def forward(self, x: Tensor) -> Tensor:
        for m in self.seq:
            x = m(x)
        return x
