import networkx as nx
import numpy as np
import torch

from dynalearn.datasets import Dataset, DegreeWeightedDataset, StrengthWeightedDataset
from dynalearn.config import Config
from dynalearn.utilities import from_nary
from dynalearn.utilities import to_edge_index, onehot


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
        w = torch.FloatTensor(self.weights[i][j])
        w[w > 0] = w[w > 0] ** (-self.bias)
        w /= w.sum()
        return (x, g), y, w


class DegreeWeightedDiscreteDataset(DiscreteDataset, DegreeWeightedDataset):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        DiscreteDataset.__init__(self, config)
        DegreeWeightedDataset.__init__(self, config)


class StrengthWeightedDiscreteDataset(DiscreteDataset, StrengthWeightedDataset):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        DiscreteDataset.__init__(self, config)
        DegreeWeightedDataset.__init__(self, config)


class StateWeightedDiscreteDataset(DiscreteDataset):
    def _get_counts_(self, data):
        counts = {}
        degrees = []
        if self.window_size > self.threshold_window_size:
            window_size = self.threshold_window_size
        else:
            window_size = self.window_size
        eff_num_states = self.num_states ** window_size
        for i in range(data["networks"].size):
            g = data["networks"][i].data
            adj = nx.to_numpy_array(g)
            for j in range(data["inputs"][i].size):
                s = np.array(
                    [
                        from_nary(ss[-window_size:], base=self.num_states)
                        for ss in data["inputs"][i].get(j)
                    ]
                )
                ns = np.zeros((s.shape[0], eff_num_states))
                for k in range(eff_num_states):
                    ns[:, k] = adj @ (s == k)

                for k in range(s.shape[0]):
                    ss = (s[k], *ns[k])
                    if ss in counts:
                        counts[ss] += 1
                    else:
                        counts[ss] = 1
        return counts

    def _get_weights_(self, data):
        weights = {}
        counts = self._get_counts_(data)
        if self.window_size > self.threshold_window_size:
            window_size = self.threshold_window_size
        else:
            window_size = self.window_size
        eff_num_states = self.num_states ** window_size
        for i in range(data["networks"].size):
            g = data["networks"][i].data
            weights[i] = np.zeros((data["inputs"][i].size, g.number_of_nodes()))
            adj = nx.to_numpy_array(g)
            for j in range(data["inputs"][i].size):
                s = np.array(
                    [
                        from_nary(ss[-window_size:], base=self.num_states)
                        for ss in data["inputs"][i].get(j)
                    ]
                )

                ns = np.zeros((s.shape[0], eff_num_states))
                for k in range(eff_num_states):
                    ns[:, k] = adj @ (s == k)

                for k in range(s.shape[0]):
                    ss = (s[k], *ns[k])
                    weights[i][j, k] = counts[ss]

        return weights
