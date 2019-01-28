"""

dataset.py

Created by Charles Murphy on 07-09-18.
Copyright © 2018 Charles Murphy. All rights reserved.
Quebec, Canada

Defines Dynamical_Network_Dataset classes for dynamics on networks.

"""

import torch
from torch.utils.data import Sampler
import numpy as np


class Random_Sampler_with_length(Sampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, length):
        self.length = length
        self.data_source = data_source

    def __iter__(self):
        return iter(torch.randint(0, len(self.data_source), self.length).tolist())

    def __len__(self):
        return self.length


def random_split(dataset, val_size):
    index = set(range(len(dataset)))
    val_index = list(np.random.choice(list(index), int(val_size * len(dataset))))
    train_index = list(index.difference(set(val_index)))

    return Subset(dataset, train_index), Subset(dataset, val_index)