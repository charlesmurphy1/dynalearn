import numpy as np
import torch

from dynalearn.datasets import Dataset, DegreeWeightedDataset
from dynalearn.config import Config

class MarkovDataset(Dataset):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        Dataset.__init__(self, config)

    def _generate_sequence_(self, m_network, m_dynamics, details, pb=None):
        networks = {}
        inputs = {}
        targets = {}
        gt_targets = {}

        for i in range(details.num_networks):
            g = m_network.generate()
            networks[i] = g
            m_dynamics.network = g
            x = m_dynamics.initial_state()
            inputs[i] = np.zeros((details.num_samples, m_network.num_nodes))
            targets[i] = np.zeros((details.num_samples, m_network.num_nodes))
            gt_targets[i] = np.zeros(
                (details.num_samples, m_network.num_nodes, self.num_states)
            )
            for j in range(details.num_samples):
                inputs[i][j] = x
                targets[i][j] = m_dynamics.sample(x)
                gt_targets[i][j] = m_dynamics.predict(x)
                if j % self.resampling_time == 0:
                    x = m_dynamics.initial_state()
                else:
                    x = y
                if experiment.verbose == 1:
                    pb.update()
        data = {}
        data["inputs"] = inputs
        data["targets"] = targets
        data["gt_targets"] = gt_targets
        data["networks"] = networks
        return data

    def __getitem__(self, index):
        i, j = self.indices[index]
        edge_index = to_edge_index(self.networks[i])
        x = torch.FloatTensor(self.inputs[i][j])
        if self.with_truth:
            y = torch.FloatTensor(self.gt_targets[i][j])
        else:
            y = torch.LongTensor(self.targets[i][j])
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
