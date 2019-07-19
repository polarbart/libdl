import os
from skimage import io
from torch.utils.data import Dataset
import torch
import pickle


class DistractedDriver(Dataset):
    def __init__(self, path, val=False, transforms=None):
        with open(os.path.join(path, 'lval' if val else 'ltrain'), 'rb') as f:
            self.data = pickle.load(f)
        self.path = path
        self.transforms = transforms

    def __getitem__(self, i):
        n, c = self.data[i]
        img = io.imread(os.path.join(self.path, 'train', f'c{c}', n))
        if self.transforms is not None:
            img = self.transforms(img)
        return img, torch.tensor(c)

    def __len__(self):
        return len(self.data)
