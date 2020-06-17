import numpy as np
import torch

from dynalearn.datasets import MarkovDataset, DegreeWeightedDataset
from dynalearn.config import Config
from dynalearn.utilities import from_nary


class NonMarkovDataset(MarkovDataset):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        self.window = config.window
        Dataset.__init__(self, config)

    def __getitem__(self, index):
        i, j = self.indices[index]
        edge_index = to_edge_index(self.networks[i])
        x = torch.FloatTensor(self.inputs[i][j : j + self.window])
        if self.with_truth:
            y = torch.FloatTensor(self.gt_targets[i][j + self.window])
        else:
            y = torch.LongTensor(self.targets[i][j + self.window])
        w = torch.FloatTensor(self.weights[i][j + self.window])
        w[w > 0] = w[w > 0] ** (-self.bias)
        w /= w.sum()
        return (x, edge_index), y, w


class DegreeWeightedNonMarkovDataset(NonMarkovDataset, DegreeWeightedDataset):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        NonMarkovDataset.__init__(self, config)
        DegreeWeightedDataset.__init__(self, config)


class StateWeightedNonMarkovDataset(MarkovNonDataset):
    def _get_counts_(self):
        counts = {}
        degrees = []
        eff_num_states = self.num_states ** self.window
        for i, g in self.networks.items():
            adj = nx.to_numpy_array(g)
            for j in range(self.inputs[i].shape[0] - self.window):
                s = np.array(
                    [
                        from_nary(ss, self.num_states)
                        for ss in self.inputs[i][j : j + self.window]
                    ]
                )
                ns = np.zeros((s.shape[1], eff_num_states))
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
        eff_num_states = self.num_states ** self.window

        for i, g in self.networks.items():
            weights[i] = np.zeros(self.inputs[i].shape)
            adj = nx.to_numpy_array(g)
            for j in range(self.inputs[i].shape[0] - self.window):
                s = np.array(
                    [
                        from_nary(ss, self.num_states)
                        for ss in self.inputs[i][j : j + self.window]
                    ]
                )
                ns = np.zeros((s.shape[1], eff_num_states))
                for k in range(eff_num_states):
                    ns[:, k] = adj @ (s == k)

                for k in range(s.shape[0]):
                    ss = (s[k], *ns[k])
                    weights[i][j, k] = counts[ss]

        return weights
