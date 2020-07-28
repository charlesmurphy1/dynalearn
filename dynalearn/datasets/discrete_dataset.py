import networkx as nx
import numpy as np
import torch

from dynalearn.datasets import Dataset, DegreeWeightedDataset
from dynalearn.config import Config
from dynalearn.utilities import from_nary
from dynalearn.utilities import to_edge_index, onehot


class DiscreteDataset(Dataset):
    def __getitem__(self, index):
        i, j = self.indices[index]
        edge_index = self.networks[i]
        x = torch.FloatTensor(self.inputs[i][j])
        if len(self.targets[i][j].shape) == 1:
            y = onehot(self.targets[i][j], num_class=self.num_states)
        else:
            y = self.targets[i][j]
        y = torch.FloatTensor(y)
        w = torch.FloatTensor(self.weights[i][j])
        w[w > 0] = w[w > 0] ** (-self.bias)
        w /= w.sum()
        return (x, edge_index), y, w


class DegreeWeightedDiscreteDataset(DiscreteDataset, DegreeWeightedDataset):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        DiscreteDataset.__init__(self, config)
        DegreeWeightedDataset.__init__(self, config)


class StateWeightedDiscreteDataset(DiscreteDataset):
    def _get_counts_(self):
        counts = {}
        degrees = []
        if self.window_size > self.threshold_window_size:
            window_size = self.threshold_window_size
        else:
            window_size = self.window_size
        eff_num_states = self.num_states ** window_size
        for i in range(self.networks.size):
            g = self.networks.data[i]
            adj = nx.to_numpy_array(g)
            for j in range(self.inputs[i].size):
                s = np.array(
                    [
                        from_nary(ss[:window_size], base=self.num_states)
                        for ss in self.inputs[i][j].T
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

    def _get_weights_(self):
        weights = {}
        counts = self._get_counts_()
        if self.window_size > self.threshold_window_size:
            window_size = self.threshold_window_size
        else:
            window_size = self.window_size
        eff_num_states = self.num_states ** window_size
        for i in range(self.networks.size):
            g = self.networks.data[i]
            weights[i] = np.zeros((self.inputs[i].size, *self.inputs[i].shape[1:]))
            adj = nx.to_numpy_array(g)
            for j in range(self.inputs[i].size):
                s = np.array(
                    [
                        from_nary(ss[:window_size], base=self.num_states)
                        for ss in self.inputs[i][j].T
                    ]
                )
                ns = np.zeros((s.shape[0], eff_num_states))
                for k in range(eff_num_states):
                    ns[:, k] = adj @ (s == k)

                for k in range(s.shape[0]):
                    ss = (s[k], *ns[k])
                    weights[i][j, k] = counts[ss]

        return weights