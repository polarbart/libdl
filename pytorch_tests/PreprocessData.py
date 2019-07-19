import os
import pickle
import random

random.seed(0)

path = '/home/superbabes/Downloads/ddriver'

files = [[] for _ in range(10)]

with open(os.path.join(path, 'driver_imgs_list.csv')) as f:
    f.readline()
    for l in f.readlines():
        l = l.split(',')
        n = l[2].strip('\n')  # filename
        c = int(l[1][1:])  # class
        files[c].append((n, c))

for f in files:
    random.shuffle(f)

val = [i for f in files for i in f[:128]]
train = [i for f in files for i in f[128:]]

with open(os.path.join(path, 'lval'), 'wb') as f:
    pickle.dump(val, f)
with open(os.path.join(path, 'ltrain'), 'wb') as f:
    pickle.dump(train, f)



