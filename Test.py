import numpy as np
import libdl
l = libdl
import timeit
import torch, torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
print(dir(libdl))
np.random.seed(1)

x = libdl.Tensor4(np.asfortranarray(np.random.rand(14, 14, 32, 64).astype(np.float32)), True)
f = libdl.Tensor4(np.asfortranarray(np.random.rand(5, 5, 32, 64).astype(np.float32)), True)
b = libdl.Tensor1(np.asfortranarray(np.random.rand(64).astype(np.float32)), True)
print(timeit.timeit(lambda: libdl.conv_2d(x, f, b, 2), number=1))



tx = torch.tensor(x.numpy().transpose(), requires_grad=True)
tf = torch.tensor(f.numpy().transpose(), requires_grad=True)
tb = torch.tensor(b.numpy(), requires_grad=True)
print(timeit.timeit(lambda: F.conv2d(tx, tf, tb, padding=2), number=1))



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


channels = 10
t = libdl.Tensor4(np.asfortranarray(np.random.rand(12, channels, 8, 5).astype(np.float32).transpose()), True)
g = libdl.Tensor1(np.ones(channels, dtype=np.float32), True)
b = libdl.Tensor1(np.zeros(channels, dtype=np.float32), True)
m = libdl.Tensor1(np.ones(channels, dtype=np.float32), False)
v = libdl.Tensor1(np.zeros(channels, dtype=np.float32), False)
r = libdl.batchNorm2d(t, g, b, m, v, .1, 1e-8, False)

tt = torch.tensor(t.numpy().transpose(), requires_grad=True)
tg = torch.tensor(g.numpy().transpose(), requires_grad=True)
tb = torch.tensor(b.numpy().transpose(), requires_grad=True)
tm = torch.tensor(m.numpy().transpose(), requires_grad=False)
tv = torch.tensor(v.numpy().transpose(), requires_grad=False)

tr = F.batch_norm(tt, tm, tv, tg, tb, training=True, eps=1e-8)
print(np.allclose(tr.detach().numpy(), r.numpy().transpose(), atol=1e-5))

libdl.sum(r).backward()
torch.sum(tr).backward()

print(np.allclose(tt.detach().numpy(), t.numpy().transpose(), atol=1e-8))
print(np.allclose(tg.detach().numpy(), g.numpy().transpose(), atol=1e-8))
print(np.allclose(tb.detach().numpy(), b.numpy().transpose(), atol=1e-8))





channels = 10
t = libdl.Tensor4(np.asfortranarray(np.random.rand(12, channels, 8, 5).astype(np.float32).transpose()), True)
g = libdl.Tensor1(np.ones(channels, dtype=np.float32), True)
b = libdl.Tensor1(np.zeros(channels, dtype=np.float32), True)
m = libdl.Tensor1(np.ones(channels, dtype=np.float32), False)
v = libdl.Tensor1(np.zeros(channels, dtype=np.float32), False)
r = libdl.batchNorm2d(t, g, b, m, v, .1, 1e-8, False)

tt = torch.tensor(t.numpy().transpose(), requires_grad=True)
tg = torch.tensor(g.numpy().transpose(), requires_grad=True)
tb = torch.tensor(b.numpy().transpose(), requires_grad=True)
tm = torch.tensor(m.numpy().transpose(), requires_grad=False)
tv = torch.tensor(v.numpy().transpose(), requires_grad=False)

tr = F.batch_norm(tt, tm, tv, tg, tb, training=True, eps=1e-8)
print(np.allclose(tr.detach().numpy(), r.numpy().transpose(), atol=1e-5))

libdl.sum(r).backward()
torch.sum(tr).backward()

print(np.allclose(tt.detach().numpy(), t.numpy().transpose(), atol=1e-8))
print(np.allclose(tg.detach().numpy(), g.numpy().transpose(), atol=1e-8))
print(np.allclose(tb.detach().numpy(), b.numpy().transpose(), atol=1e-8))
'''