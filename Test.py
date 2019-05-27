import numpy as np
import libdl
print(dir(libdl))


t = libdl.Tensor1(np.asfortranarray(np.array([1, 2, 3]).astype(np.float32)), True)
a = (t + t) + t
a.backward()
print(t.grad())
