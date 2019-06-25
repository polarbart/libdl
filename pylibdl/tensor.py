import numpy as np
import libdl


def tensor(data, requires_grad=False, dtype=np.float32):

    supported_datatypes = [np.float32]
    if dtype not in supported_datatypes:
        raise RuntimeError(f'{dtype} is currently not supported. Only {supported_datatypes} are supported.')

    supported_dims = 4
    if len(data.shape) > supported_dims:
        raise RuntimeError(f'Only tensors up to {supported_dims} are supported. Got {len(data.shape)} dimensions.')

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

