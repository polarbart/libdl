import numpy as np
import pylibdl as libdl
from typing import Union

_tensor_types = [libdl.Tensor0, libdl.Tensor1, libdl.Tensor2, libdl.Tensor3, libdl.Tensor4]
_supported_datatypes = [np.float32]
Tensor = Union[libdl.Tensor0, libdl.Tensor1, libdl.Tensor2, libdl.Tensor3, libdl.Tensor4]


def tensor(data: np.ndarray, requires_grad: bool = False, dtype: np.dtype = np.float32) -> Tensor:
    """
    turns a numpy array into a libdl tensor

    :param data: the numpy array
    :param requires_grad: if the tensor requires a gradient
    :param dtype: the type of the tensor
    :return: a tensor with the data of data
    """
    _check_input(len(data.shape), dtype)

    data = np.asfortranarray(data.astype(dtype))
    if len(data.shape) == 0:
        return libdl.Tensor0(data, requires_grad)
    elif len(data.shape) == 1:
        return libdl.Tensor1(data, requires_grad)
    elif len(data.shape) == 2:
        return libdl.Tensor2(data, requires_grad)
    elif len(data.shape) == 3:
        return libdl.Tensor3(data, requires_grad)
    elif len(data.shape) == 4:
        return libdl.Tensor4(data, requires_grad)


def _check_input(dimensions: int, dtype: np.dtype):
    if dtype not in _supported_datatypes:
        raise RuntimeError(f'{dtype} is currently not supported. Only {_supported_datatypes} are supported.')

    supported_dims = 4
    if dimensions > supported_dims:
        raise RuntimeError(f'Only tensors up to {supported_dims} dimensions are supported. Got {dimensions} dimensions.')



