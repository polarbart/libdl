import numpy as np
import os
import warnings
from skimage.io import imread
from skimage.transform import resize
from pylibdl.data import Dataset, DataLoader
from pylibdl.modules import *
import random

warnings.filterwarnings('ignore')


def validate(model, val_data, batch_size=128):
    val_loader = DataLoader(val_data, batch_size, shuffle=False, drop_last=False)
    is_train = model.is_train
    model.eval()
    accs = []
    with libdl.no_grad():
        for imgs, labels in val_loader:
            pred = model(imgs)
            accs.append((np.argmax(pred.data, axis=0) == np.argmax(labels.data, axis=0)).mean())
    model.train(is_train)
    return np.mean(accs)


class DistractedDriver(Dataset):

    label_names = [
        'safe driving',
        'texting - right',
        'talking on the phone - right',
        'texting - left',
        'talking on the phone - left',
        'operating the radio',
        'drinking',
        'reaching behind',
        'hair and makeup',
        'talking to passenger'
    ]

    mean = np.array([0.3142706, 0.3803778, 0.37310703]).reshape((3, 1, 1))
    std = np.array([0.29078071, 0.33230701, 0.33428186]).reshape((3, 1, 1))

    @staticmethod
    def preprocess_data(path):
        random.seed(0)
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

    def __init__(self, path, val=False, size=(128, 128)):

        if not os.path.exists(os.path.join(path, 'lval')):
            DistractedDriver.preprocess_data(path)

        with open(os.path.join(path, 'lval' if val else 'ltrain'), 'rb') as f:
            self.data = pickle.load(f)

        self.path = path
        self.size = size

    @staticmethod
    def normalize(x):
        return (x - DistractedDriver.mean) / DistractedDriver.std

    @staticmethod
    def denormalize(x):
        return x * DistractedDriver.std + DistractedDriver.mean

    def __getitem__(self, i):
        n, c = self.data[i]
        img = imread(os.path.join(self.path, 'train', f'c{c}', n))
        img = resize(img, self.size, anti_aliasing=False)
        lc = np.zeros(10)
        lc[c] = 1
        return self.normalize(img.transpose((2, 0, 1))), lc

    def __len__(self):
        return len(self.data)
