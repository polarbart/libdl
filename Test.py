import numpy as np
import libdl
print(dir(libdl))

a = libdl.Tensor3(np.asfortranarray(np.arange(64, dtype=np.float32).reshape((4, 4, 4))), False)
x = libdl.Tensor2(np.asfortranarray(np.arange(16, dtype=np.float32).reshape((4, 4))), False)
b = libdl.matmul(a, x)
print(b.data)
