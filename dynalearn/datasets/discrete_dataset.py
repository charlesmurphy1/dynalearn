import networkx as nx
import numpy as np
import torch

from dynalearn.datasets import Dataset, StructureWeightDataset
from dynalearn.config import Config
from dynalearn.utilities import from_nary
from dynalearn.utilities import to_edge_index, onehot
from dynalearn.datasets.weights import DiscreteStateWeight, DiscreteCompoundStateWeight


class DiscreteDataset(Dataset):
    def __getitem__(self, index):
        i, j = self.indices[index]
        g = self.networks[i].get()
        x = torch.FloatTensor(self.inputs[i].get(j))
        if len(self.targets[i].get(j).shape) == 1:
            y = onehot(self.targets[i].get(j), num_class=self.num_states)
        else:
            y = self.targets[i].get(j)
        y = torch.FloatTensor(y)
        w = torch.FloatTensor(self.weights[i].get(j))
        w[w > 0] = w[w > 0] ** (-self.bias)
        w /= w.sum()
        return (x, g), y, w


class DiscreteStructureWeightDataset(DiscreteDataset, StructureWeightDataset):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        DiscreteDataset.__init__(self, config)
        StructureWeightDataset.__init__(self, config)


class DiscreteStateWeightDataset(DiscreteDataset):
    def _get_weights_(self, data):
        if self.config.compounded:
            weights = DiscreteCompoundStateWeight()
        else:
            weights = DiscreteStateWeight()
        weights.compute(self, verbose=self.verbose)
        return weights
