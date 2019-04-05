import torch
from torch.utils.data.dataset import Dataset
import numpy as np


class Normal_Dataset(Dataset):
    """docstring for Normal_Dataset"""
    def __init__(self, numsample=1000, dim=1, means=None, stds=None):
        self.dim = dim
        if means is None:
            self.means = np.random.normal(0, 1, self.dim)
        else:
            self.means = means

        if stds is None:
            self.stds = abs(1 + np.random.normal(loc=0, scale=1, size=self.dim))
        else:
            self.stds = stds

        self.data = []
        self._generate_data(numsample)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def _generate_data(self, numsample):
        for i in range(numsample):
            self.data.append(np.random.normal(loc=self.means, scale=self.stds))

        return None