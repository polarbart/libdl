import numpy as np
import libdl

a = libdl.Tensor(np.asfortranarray(np.arange(16, dtype=np.float32).reshape((4, 4))))
b = libdl.add(a, a)

print(b.data)
