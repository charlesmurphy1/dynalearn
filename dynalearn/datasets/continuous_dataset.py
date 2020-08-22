import numpy as np
import networkx as nx
import torch
import tqdm

from scipy.stats import gaussian_kde
from dynalearn.datasets import Dataset, DegreeWeightedDataset, StrengthWeightedDataset
from dynalearn.config import Config
from dynalearn.utilities import from_nary
from dynalearn.utilities import to_edge_index, onehot


class ContinuousDataset(Dataset):
    def __getitem__(self, index):
        i, j = self.indices[index]
        g = self.networks[i].get()
        x = torch.FloatTensor(self.inputs[i].get(j))
        y = torch.FloatTensor(self.targets[i].get(j))
        w = torch.FloatTensor(self.weights[i][j])
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

    def _get_distributions_(self, data, pb=None):
        samples = {}
        com_states = {}
        degree_count = {}

        for i in range(data["networks"].size):
            g = data["networks"][i].data
            for node, degree in g.degree():
                if degree in degree_count:
                    degree_count[degree] += 1
                else:
                    degree_count[degree] = 1

            for j in range(data["inputs"][i].size):
                if pb is not None:
                    pb.update()
                s = data["inputs"][i].get(j)
                for k, ss in enumerate(s):
                    degree = g.degree(k)
                    com_states[(i, j, k)] = np.array(
                        [
                            np.concatenate([ss.reshape(-1), s[l].reshape(-1)])
                            for l in g.neighbors(k)
                        ]
                    )
                    if degree not in samples:
                        samples[degree] = [
                            com_states[(i, j, k)][l]
                            for l in range(com_states[(i, j, k)].shape[0])
                        ]
                    elif len(samples[degree]) > self.max_num_points:
                        pass
                    else:
                        samples[degree].extend(
                            [
                                com_states[(i, j, k)][l]
                                for l in range(com_states[(i, j, k)].shape[0])
                            ]
                        )
        kde_dict = {}
        degree_dict = {}
        mean_dict = {}
        std_dict = {}
        z = np.sum(list(degree_count.values()))
        for k, s in samples.items():
            s = np.array(s).T
            mean_dict[k] = np.mean(s)
            std_dict[k] = np.std(s)
            s = (s - np.mean(s)) / np.std(s)
            kde_dict[k] = gaussian_kde(s)
            degree_dict[k] = degree_count[k] / z
        return kde_dict, degree_dict, com_states, mean_dict, std_dict

    def _get_weights_(self, data):
        if self.verbose == 1:
            pb = tqdm.tqdm(
                range(
                    np.sum(
                        [
                            2 * data["inputs"][i].size
                            for i in range(data["networks"].size)
                        ]
                    )
                ),
                "Compute the weights",
            )
        else:
            pb = None
        weights = {}
        (
            kde_dict,
            degree_dict,
            com_states,
            mean_dict,
            std_dict,
        ) = self._get_distributions_(data, pb)

        for i in range(data["networks"].size):
            g = data["networks"][i].data
            weights[i] = np.zeros((data["inputs"][i].size, g.number_of_nodes()))
            for j in range(data["inputs"][i].size):
                if pb is not None:
                    pb.update()
                s = data["inputs"][i].get(j)
                for k, ss in enumerate(s):
                    d = g.degree(k)
                    cs = ((com_states[(i, j, k)] - mean_dict[d]) / std_dict[d]).T
                    weights[i][j, k] = degree_dict[d] * np.prod(kde_dict[d](cs)) ** (
                        1.0 / d
                    )
        if pb is not None:
            pb.close()

        return weights
