import numpy as np
import libdl
l = libdl
import timeit
import torch, torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
print(dir(libdl))
np.random.seed(0)
def fun():
    x = libdl.Tensor4(np.asfortranarray(np.random.rand(20, 20, 16, 64).astype(np.float32)), True)
    f = libdl.Tensor4(np.asfortranarray(np.random.rand(5, 5, 16, 32).astype(np.float32)), True)
    b = libdl.Tensor1(np.asfortranarray(np.random.rand(32).astype(np.float32)), True)
    r = libdl.conv2d(x, f, b, 2)
    # r2 = libdl.maxpool2d(r, 2)
    print(r.numpy().shape)
# print(timeit.timeit(fun, number=1))


t = libdl.Tensor4(np.asfortranarray(np.random.rand(7, 7, 32, 64)).astype(np.float32), False)
print(libdl.maxpool2d(t, 2).numpy().shape)
print(F.max_pool2d(torch.tensor(np.random.rand(7, 7, 32, 64).transpose()), 2, 2).numpy().shape)
exit()


x = libdl.Tensor4(np.asfortranarray(np.random.rand(10, 10, 16, 64).astype(np.float32)), True)
f = libdl.Tensor4(np.asfortranarray(np.random.rand(5, 5, 16, 32).astype(np.float32)), True)
b = libdl.Tensor1(np.asfortranarray(np.random.rand(32).astype(np.float32)), True)
r = libdl.conv2d(x, f, b, 0)
print(r.numpy().shape)
tx = torch.tensor(x.numpy().transpose(), requires_grad=True)
tf = torch.tensor(f.numpy().transpose(), requires_grad=True)
tb = torch.tensor(b.numpy(), requires_grad=True)
tr = F.conv2d(tx, tf, tb, padding=0)
print(np.allclose(r.numpy(), tr.detach().numpy().transpose()))
libdl.sum(r).backward()
torch.sum(tr).backward()
print(np.allclose(x.grad(), tx.grad.numpy().transpose()))
print(np.allclose(f.grad(), tf.grad.numpy().transpose()))
print(np.allclose(b.grad(), tb.grad.numpy()))

#print(f.grad().transpose())
#print(tf.grad.numpy())

'''
x = libdl.Tensor4(np.asfortranarray(np.random.rand(4, 4, 1, 1).astype(np.float32)), True)
f = libdl.Tensor4(np.asfortranarray(np.random.rand(3, 4, 1, 2).astype(np.float32)), True)
b = libdl.Tensor1(np.asfortranarray(np.random.rand(2).astype(np.float32)), True)
r = libdl.conv2d(x, f, b, 0)
tx = torch.tensor(x.numpy().transpose(), requires_grad=True)
tf = torch.tensor(f.numpy().transpose(), requires_grad=True)
tb = torch.tensor(b.numpy(), requires_grad=True)
tr = F.conv2d(tx, tf, tb, padding=0)
print(np.allclose(r.numpy(), tr.detach().numpy().transpose()))
libdl.sum(r).backward()
torch.sum(tr).backward()
print(np.allclose(x.grad(), tx.grad.numpy().transpose()))
print(np.allclose(f.grad(), tf.grad.numpy().transpose()))
print(np.allclose(b.grad(), tb.grad.numpy()))
print(x.grad().transpose())
print(tx.grad.numpy())



x = libdl.Tensor4(np.asfortranarray(np.random.rand(8, 8, 4, 5).astype(np.float32)), True)
tx = torch.tensor(x.numpy().transpose(), requires_grad=True)
r = libdl.maxpool2d(x, 3)
tr = F.max_pool2d(tx, 3, 3)
print(np.allclose(r.numpy(), tr.detach().numpy().transpose()))
libdl.sum(r).backward()
torch.sum(tr).backward()
print(np.allclose(x.grad(), tx.grad.numpy().transpose()))
#print(x.numpy().transpose())
print()
#print(r.numpy().transpose())
print()
#print(tr.detach().numpy())
exit()


y = np.random.rand(10, 4).argmax(0)
ty = torch.tensor(y)

x = libdl.Tensor2(np.asfortranarray(np.random.rand(10, 4).astype(np.float32)), True)
y = libdl.Tensor2(np.eye(10)[y].transpose().astype(np.float32), True)
r = libdl.crossEntropyWithLogits(x, y)
tx = torch.tensor(x.numpy().transpose(), requires_grad=True)
tr = F.cross_entropy(tx, ty)
print(np.allclose(r.numpy(), tr.detach().numpy().transpose()))
r.backward()
tr.backward()
print(np.allclose(x.grad(), tx.grad.numpy().transpose()))
exit()
'''