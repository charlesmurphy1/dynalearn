import numpy as np
import networkx as nx
import torch
import tqdm

from scipy.stats import gaussian_kde
from dynalearn.datasets import Dataset, StructureWeightDataset
from dynalearn.datasets.weights import (
    ContinuousStateWeight,
    ContinuousCompoundStateWeight,
    StrengthContinuousStateWeight,
    StrengthContinuousCompoundStateWeight,
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


class ContinuousStructureWeightDataset(ContinuousDataset, StructureWeightDataset):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        ContinuousDataset.__init__(self, config)
        StructureWeightDataset.__init__(self, config)


class ContinuousStateWeightDataset(ContinuousDataset):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        ContinuousDataset.__init__(self, config)
        self.max_num_points = config.max_num_points
        self.compounded = config.compounded
        self.reduce = config.reduce

    def _get_weights_(self, data):
        if self.m_networks.is_weighted and self.compounded:
            weights = StrengthContinuousCompoundStateWeight(reduce=self.reduce)
        elif self.m_networks.is_weighted and not self.compounded:
            weights = StrengthContinuousStateWeight(reduce=self.reduce)
        elif not self.m_networks.is_weighted and self.compounded:
            weights = ContinuousCompoundStateWeight(reduce=self.reduce)
        else:
            weights = ContinuousStateWeight()
        weights.compute(self, verbose=self.verbose)
        return weights
