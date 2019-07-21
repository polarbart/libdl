import numpy as np
import libdl

_tensor_types = [libdl.Tensor0, libdl.Tensor1, libdl.Tensor2, libdl.Tensor3, libdl.Tensor4]
_supported_datatypes = [np.float32]


def tensor(data, requires_grad=False, dtype=np.float32):

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


def zeros(shape, requires_grad=False, dtype=np.float32):
    return constant(shape, 0., requires_grad, dtype)


def ones(shape, requires_grad=False, dtype=np.float32):
    return constant(shape, 1., requires_grad, dtype)


def constant(shape, c, requires_grad=False, dtype=np.float32):
    _check_input(len(shape), dtype)
    if len(shape) == 0:
        return libdl.constant(shape, c, requires_grad)
    elif len(shape) == 1:
        return libdl.constant(shape, c, requires_grad)
    elif len(shape) == 2:
        return libdl.constant(shape, c, requires_grad)
    elif len(shape) == 3:
        return libdl.constant(shape, c, requires_grad)
    elif len(shape) == 4:
        return libdl.constant(shape, c, requires_grad)


def _check_input(dimensions, dtype):
    if dtype not in _supported_datatypes:
        raise RuntimeError(f'{dtype} is currently not supported. Only {_supported_datatypes} are supported.')

    supported_dims = 4
    if dimensions > supported_dims:
        raise RuntimeError(f'Only tensors up to {supported_dims} dimensions are supported. Got {dimensions} dimensions.')
