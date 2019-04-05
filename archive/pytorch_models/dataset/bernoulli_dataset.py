import torch
from torch.utils.data.dataset import Dataset
import numpy as np


class Bernoulli_Dataset(Dataset):
    """docstring for Bernoulli_Dataset"""
    def __init__(self, numsample=1000, dim=1, p=None):
        self.dim = dim
        if p is None:
            self.p = np.random.rand(self.dim)
        else:
            self.p = p

        self.data = []
        self._generate_data(numsample)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def _generate_data(self, numsample):
        for i in range(numsample):
            x = np.zeros(self.dim)
            r = np.random.rand(self.dim)
            x[r < self.p] = 1
            x = torch.Tensor(x)
            self.data.append(x)

        return None