import numpy as np
import torch

from dynalearn.datasets import Dataset, DegreeWeightedDataset
from dynalearn.config import Config
from dynalearn.utilities import from_nary
from dynalearn.utilities import to_edge_index, onehot


class ContinuousDataset(Dataset):
    def __getitem__(self, index):
        i, j = self.indices[index]
        edge_index = to_edge_index(self.networks[i])
        x, y = self.normalize(
            self.inputs[i][
                j - (self.window_size - 1) * self.window_step : j + 1 : self.window_step
            ],
            self.targets[i][j],
        )
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        w = torch.FloatTensor(self.weights[i][j])
        w[w > 0] = w[w > 0] ** (-self.bias)
        w /= w.sum()
        return (x, edge_index), y, w


class DegreeWeightedContinuousDataset(ContinuousDataset, DegreeWeightedDataset):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        ContinuousDataset.__init__(self, config)
        DegreeWeightedDataset.__init__(self, config)


class StateWeightedContinuousDataset(ContinuousDataset):
    def _get_distributions_(self):
        samples = {}
        degree_count = {}
        for i, g in self.networks.items():
            adj = nx.to_numpy_array(g)
            for degree in adj.sum(0):
                if degree in degree_count:
                    degree_count[degree] += 1
                else:
                    degree_count[degree] = 0

            for j, s in enumerate(self.inputs[i]):
                for k, ss in enumerate(s):
                    degree = adj[k].sum()
                    ns = np.array([s[l] for l in np.where(adj[k] == 1)[0]])
                    if len(samples[degree]) > 1e4:
                        pass
                    elif degree not in samples:
                        samples[degree] = [
                            np.concatenate([ss.reshape(-1), _ns.reshape(-1)])
                            for _ns in ns
                        ]
                    else:
                        samples[degree].extend(
                            [
                                np.concatenate([ss.reshape(-1), _ns.reshape(-1)])
                                for _ns in ns
                            ]
                        )
        kde_dict = {}
        degree_dict = {}
        z = np.sum(list(degree_count.values()))
        for k, s in samples.items():
            s = np.array(s).T
            kde_dict[k] = gaussian_kde(s)
            degree_dict[k] = degree_count[k] / z
        return kde_dict, degree_dict

    def _get_weights_(self):
        weights = {}
        kde_dict, degree_dict = self._get_distributions_()

        for i, g in self.networks.items():
            weights[i] = np.zeros(self.inputs[i].shape)
            adj = nx.to_numpy_array(g)
            for j, s in enumerate(self.inputs[i]):
                for k, ss in enumerate(s):
                    degree = adj[k].sum()
                    weights[i][j, k] = degree_dict[degree] * np.prod(
                        [
                            kde_dict[degree](
                                np.concatenate([ss.reshape(-1), s[l].reshape(-1)])
                            )
                            for l in np.where(adj[k] == 1)[0]
                        ]
                    )

        return weights
