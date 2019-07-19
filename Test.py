import numpy as np
import libdl

l = libdl
import timeit
import torch, torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader

print(dir(libdl))
np.random.seed(1)


channels = 1
t = libdl.Tensor4(np.asfortranarray(np.random.rand(channels, 4, 4, 1).astype(np.float32)), True)
g = libdl.Tensor1(np.ones(channels, dtype=np.float32), True)
b = libdl.Tensor1(np.zeros(channels, dtype=np.float32), True)
m = libdl.Tensor1(np.ones(channels, dtype=np.float32), False)
v = libdl.Tensor1(np.zeros(channels, dtype=np.float32), False)
r = libdl.batch_norm_2d(t, g, b, m, v, .1, 1e-8, True)
#print(timeit.timeit(lambda: libdl.batch_norm_2d(t, g, b, m, v, .1, 1e-8, True), number=1)*1000)

tt = torch.tensor(t.numpy().transpose((3, 0, 1, 2)), requires_grad=True)
tg = torch.tensor(g.numpy(), requires_grad=True)
tb = torch.tensor(b.numpy(), requires_grad=True)
tm = torch.tensor(m.numpy(), requires_grad=False)
tv = torch.tensor(v.numpy(), requires_grad=False)

tr = F.batch_norm(tt, tm, tv, tg, tb, training=False, eps=1e-8)
#print(timeit.timeit(lambda: F.batch_norm(tt, tm, tv, tg, tb, training=False, eps=1e-8), number=1)*1000*25)
print(np.allclose(tr.detach().numpy(), r.numpy().transpose((3, 0, 1, 2)), atol=1e-6))

libdl.sum(r).backward()
torch.sum(tr).backward()

print(np.allclose(tt.grad.numpy(), t.grad().transpose((3, 0, 1, 2)), atol=1e-6))
print(np.allclose(tg.grad.numpy(), g.grad(), atol=1e-6))
print(np.allclose(tb.grad.numpy(), b.grad(), atol=1e-6))


print(tt.grad.numpy().squeeze())
print(t.grad().squeeze())



'''


w1 = libdl.Tensor2(np.asfortranarray(np.random.rand(512, 256).astype(np.float32)), True)
w2 = libdl.Tensor2(np.asfortranarray(w1.numpy().transpose().astype(np.float32)), True)
b = libdl.Tensor1(np.ones(512, dtype=np.float32), True)
x = libdl.Tensor2(np.asfortranarray(np.random.rand(256, 128).astype(np.float32)), True)

print(timeit.timeit(lambda: libdl.add(libdl.matmul(w1, x), b), number=1)*1000)
print(timeit.timeit(lambda: libdl.linear(w2, x, b), number=1)*1000)
print(timeit.timeit(lambda: libdl.add(libdl.matmul(w1, x), b), number=1)*1000)
print(timeit.timeit(lambda: libdl.add(libdl.matmul(w1, x), b), number=1)*1000)


x = libdl.Tensor4(np.asfortranarray(np.random.rand(32, 28, 28, 64).astype(np.float32)), True)
f = libdl.Tensor4(np.asfortranarray(np.random.rand(32, 5, 5, 64).astype(np.float32)), True)
b = libdl.Tensor1(np.asfortranarray(np.random.rand(64).astype(np.float32)), True)
r = libdl.conv_2d(x, f, b)
tx = torch.tensor(x.numpy().transpose((3, 0, 1, 2)), requires_grad=True)
tf = torch.tensor(f.numpy().transpose((3, 0, 1, 2)), requires_grad=True)
tb = torch.tensor(b.numpy(), requires_grad=True)
tr = F.conv2d(tx, tf, tb, padding=2)
print(np.allclose(r.numpy().transpose((3, 0, 1, 2)), tr.detach().numpy()))
libdl.sum(r).backward()
torch.sum(tr).backward()
print(np.allclose(x.grad().transpose((3, 0, 1, 2)), tx.grad.numpy()))
print(np.allclose(f.grad().transpose((3, 0, 1, 2)), tf.grad.numpy()))
print(np.allclose(b.grad(), tb.grad.numpy()))



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
r = libdl.cross_entropy_with_logits(x, y)
tx = torch.tensor(x.numpy().transpose(), requires_grad=True)
tr = F.cross_entropy(tx, ty)
print(np.allclose(r.numpy(), tr.detach().numpy().transpose()))
r.backward()
tr.backward()
print(np.allclose(x.grad(), tx.grad.numpy().transpose()))
exit()



channels = 10
t = libdl.Tensor4(np.asfortranarray(np.random.rand(channels, 14, 14, 4).astype(np.float32)), True)
g = libdl.Tensor1(np.ones(channels, dtype=np.float32), True)
b = libdl.Tensor1(np.zeros(channels, dtype=np.float32), True)
m = libdl.Tensor1(np.ones(channels, dtype=np.float32), False)
v = libdl.Tensor1(np.zeros(channels, dtype=np.float32), False)
r = libdl.batch_norm_2d(t, g, b, m, v, .1, 1e-8, False)
print(timeit.timeit(lambda: libdl.batch_norm_2d(t, g, b, m, v, .1, 1e-8, False), number=1)*1000)

tt = torch.tensor(t.numpy().transpose((3, 0, 1, 2)), requires_grad=True)
tg = torch.tensor(g.numpy(), requires_grad=True)
tb = torch.tensor(b.numpy(), requires_grad=True)
tm = torch.tensor(m.numpy(), requires_grad=False)
tv = torch.tensor(v.numpy(), requires_grad=False)

tr = F.batch_norm(tt, tm, tv, tg, tb, training=True, eps=1e-8)
print(timeit.timeit(lambda: F.batch_norm(tt, tm, tv, tg, tb, training=True, eps=1e-8), number=1)*1000*25)
print(np.allclose(tr.detach().numpy(), r.numpy().transpose((3, 0, 1, 2)), atol=1e-3))

libdl.sum(r).backward()
torch.sum(tr).backward()

print(np.allclose(tt.detach().numpy(), t.numpy().transpose((3, 0, 1, 2)), atol=1e-8))
print(np.allclose(tg.detach().numpy(), g.numpy().transpose(), atol=1e-8))
print(np.allclose(tb.detach().numpy(), b.numpy().transpose(), atol=1e-8))


x = libdl.Tensor4(np.asfortranarray(np.random.rand(32, 28, 28, 64).astype(np.float32)), True)
f = libdl.Tensor4(np.asfortranarray(np.random.rand(32, 5, 5, 64).astype(np.float32)), True)
b = libdl.Tensor1(np.asfortranarray(np.random.rand(64).astype(np.float32)), True)
r = libdl.conv_2d(x, f, b)
tx = torch.tensor(x.numpy().transpose((3, 0, 1, 2)), requires_grad=True)
tf = torch.tensor(f.numpy().transpose((3, 0, 1, 2)), requires_grad=True)
tb = torch.tensor(b.numpy(), requires_grad=True)
tr = F.conv2d(tx, tf, tb, padding=2)

print(timeit.timeit(lambda: libdl.sum(r).backward(), number=1))
print(timeit.timeit(lambda: torch.sum(tr).backward(), number=1))

print(timeit.timeit(lambda: libdl.conv_2d(x, f, b), number=1))
print(timeit.timeit(lambda: F.conv2d(tx, tf, tb, padding=2), number=1))


x = libdl.Tensor4(np.asfortranarray(np.random.rand(512, 16, 16, 64).astype(np.float32)), True)
tx = torch.tensor(x.numpy().transpose(), requires_grad=True)
print(np.allclose(libdl.mean(x, (1, 2)).numpy().transpose(), tx.mean(dim=(1, 2)).detach().numpy()))
libdl.mean(x, (1, 2)).backward()
tx.mean(dim=(1, 2)).sum().backward()
print(np.allclose(x.grad().transpose(), tx.grad.numpy()))
'''
