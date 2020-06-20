import networkx as nx
import numpy as np
import torch


from dynalearn.datasets.transforms import get as get_transforms
from dynalearn.datasets import Dataset, DegreeWeightedDataset
from dynalearn.config import Config
from dynalearn.utilities import to_edge_index, onehot


class MarkovDataset(Dataset):
    def __getitem__(self, index):
        i, j = self.indices[index]
        edge_index = to_edge_index(self.networks[i])
        x = torch.FloatTensor(self.inputs[i][j])
        if len(self.targets[i][j].shape) == 1:
            y = onehot(self.targets[i][j], self.num_states)
        else:
            y = self.targets[i][j]
        y = torch.FloatTensor(y)
        w = torch.FloatTensor(self.weights[i][j])
        w[w > 0] = w[w > 0] ** (-self.bias)
        w /= w.sum()
        return (x, edge_index), y, w


class DegreeWeightedMarkovDataset(MarkovDataset, DegreeWeightedDataset):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        MarkovDataset.__init__(self, config)
        DegreeWeightedDataset.__init__(self, config)


class StateWeightedMarkovDataset(MarkovDataset):
    def _get_counts_(self):
        counts = {}
        degrees = []
        for i, g in self.networks.items():
            adj = nx.to_numpy_array(g)
            for j, s in enumerate(self.inputs[i]):
                ns = np.zeros((s.shape[0], self.num_states))
                for k in range(self.num_states):
                    ns[:, k] = adj @ (s == k)

                for k in range(s.shape[0]):
                    ss = (s[k], *ns[k])
                    if ss in counts:
                        counts[ss] += 1
                    else:
                        counts[ss] = 1
        return counts

    def _get_weights_(self):
        weights = {}
        counts = self._get_counts_()

        for i, g in self.networks.items():
            weights[i] = np.zeros(self.inputs[i].shape)
            adj = nx.to_numpy_array(g)
            for j, s in enumerate(self.inputs[i]):
                ns = np.zeros((s.shape[0], self.num_states))
                for k in range(self.num_states):
                    ns[:, k] = adj @ (s == k)

                for k in range(s.shape[0]):
                    ss = (s[k], *ns[k])
                    weights[i][j, k] = counts[ss]

        return weights
