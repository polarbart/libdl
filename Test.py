import numpy as np
import libdl


a = libdl.Tensor(np.asfortranarray(np.arange(1024*1024*500, dtype=np.float32).reshape((1024*10, 1024*50))), True)
b = libdl.add(a, a)
del a
input("daf")
print(b.data.shape)
del b
input("hasdf")
