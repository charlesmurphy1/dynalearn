import numpy as np
import networkx as nx
import torch
import tqdm

from scipy.stats import gaussian_kde
from dynalearn.datasets import Dataset, DegreeWeightedDataset, StrengthWeightedDataset
from dynalearn.datasets.data import (
    NodeStrengthContinuousStateWeightData,
    ContinuousStateWeightData,
)
from dynalearn.config import Config
from dynalearn.utilities import from_nary
from dynalearn.utilities import to_edge_index, onehot


class ContinuousDataset(Dataset):
    def __getitem__(self, index):
        i, j = self.indices[index]
        g = self.networks[i].get()
        x = torch.FloatTensor(self.inputs[i].get(j))
        y = torch.FloatTensor(self.targets[i].get(j))
        w = torch.FloatTensor(self.weights[i].get(j))
        w[w > 0] = w[w > 0] ** (-self.bias)
        w /= w.sum()
        return (x, g), y, w


class DegreeWeightedContinuousDataset(ContinuousDataset, DegreeWeightedDataset):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        ContinuousDataset.__init__(self, config)
        DegreeWeightedDataset.__init__(self, config)


class StrengthWeightedContinuousDataset(ContinuousDataset, StrengthWeightedDataset):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        ContinuousDataset.__init__(self, config)
        StrengthWeightedDataset.__init__(self, config)


class StateWeightedContinuousDataset(ContinuousDataset):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        ContinuousDataset.__init__(self, config)
        self.max_num_points = config.max_num_points

    def _get_weights_(self, data):
        if self.m_networks.is_weighted:
            weights = NodeStrengthContinuousStateWeightData()
            weights.compute(self, verbose=self.verbose)
        else:
            weights = ContinuousStateWeightData()
            weights.compute(self, verbose=self.verbose)
        return weights
