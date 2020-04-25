import copy
import h5py
import networkx as nx
import numpy as np
import torch
import tqdm

from itertools import islice, chain
from .sampler import Sampler
from dynalearn.utilities import to_edge_index, Config


class Dataset(object):
    def __init__(self, config=None, data=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        self.config = config
        self.sampler = Sampler(self)
        self.bias = config.bias
        self.resampling_time = config.resampling_time
        self.with_truth = config.with_truth
        if data is not None:
            self.data = data
        else:
            self._data = None
            self._weights = {}
            self.num_states = None
            self.indices = {}

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

    def __len__(self):
        return np.sum([s.shape[0] for i, s in self.data["inputs"].items()])

    def __iter__(self):
        return self

    def __next__(self):
        index = self.sampler()
        return self[index]

    def _get_indices_(self):
        if self.inputs is None or self.targets is None or self.networks is None:
            return {}
        index = 0
        indices_dict = {}
        for i in self.networks.keys():
            for j in range(self.inputs[i].shape[0]):
                indices_dict[index] = (i, j)
                index += 1
        return indices_dict

    def _get_weights_(self):
        return {i: np.ones(self.inputs[i].shape) for i, g in self.networks.items()}

    def generate(self, experiment):
        m_network = experiment.networks
        m_dynamics = experiment.dynamics
        details = experiment.train_details
        self.num_states = int(m_dynamics.num_states)
        networks = {}
        inputs = {}
        targets = {}
        gt_targets = {}

        if experiment.verbose == 1:
            pb = tqdm.tqdm(range(details.num_networks * details.num_samples))
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
                    x = targets[i][j]
                if experiment.verbose == 1:
                    pb.update()
        if experiment.verbose == 1:
            pb.close()
        data = {}
        data["inputs"] = inputs
        data["targets"] = targets
        data["gt_targets"] = gt_targets
        data["networks"] = networks
        self.data = data

    def partition(self, node_fraction, bias=0):
        dataset = type(self)(self.config, self._data)
        weights = {i: np.zeros(w.shape) for i, w in self.weights.items()}
        for i in self.networks.keys():
            for j in range(self.inputs[i].shape[0]):
                index = np.where(self.weights[i][j] > 0)[0]
                n = int(node_fraction * index.shape[0])
                p = self.weights[i][j] ** (-bias)
                p /= p.sum()
                remove_nodes = np.random.choice(index, p=p, size=n, replace=False)

                weights[i][j] *= 0
                weights[i][j][remove_nodes] = self._weights[i][j][remove_nodes] * 1
                self._weights[i][j][remove_nodes] = 0
        dataset.weights = weights
        return dataset

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
        self.inputs = data["inputs"]
        self.targets = data["targets"]
        if "gt_targets" in data:
            self.gt_targets = data["gt_targets"]
        self.networks = data["networks"]
        self.num_states = (
            np.max(np.array(list(data["inputs"].values()))).astype("int") + 1
        )
        self.indices = self._get_indices_()
        self.weights = self._get_weights_()
        self.sampler.reset()

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights
        self.state_weights = {i: w.sum(-1) for i, w in self._weights.items()}
        self.network_weights = np.array(
            [w.sum(-1).sum(-1) for w in self._weights.values()]
        )

    @property
    def indices(self):
        return self._indices

    @indices.setter
    def indices(self, indices):
        self._indices = indices
        self.rev_indices = {(i, j): k for k, (i, j) in self.indices.items()}

    def to_batch(self, size):
        sourceiter = iter(self)
        while True:
            if (
                self.sampler.counter < len(self)
                and len(self.sampler.avail_networks) > 0
            ):
                batchiter = islice(sourceiter, size)
                yield chain([next(batchiter)], batchiter)
            else:
                self.sampler.reset()
                return

    def save(self, h5file, name=None):
        if type(h5file) is not h5py.File:
            raise ValueError("Dataset file format must be HDF5.")

        name = name or "data"

        for i, g in self.networks.items():
            if len(g.edges()) > 0:
                edge_list = to_edge_index(g)
            else:
                edge_list = np.zeros((0, 2)).astype("int")

            inputs = self.inputs[i]
            targets = self.targets[i]
            gt_targets = self.gt_targets[i]

            if f"{name}{i}" in h5file:
                del h5file[f"{name}{i}"]
            group = h5file.create_group(f"{name}{i}")
            group.create_dataset("edge_list", data=edge_list)
            group.create_dataset("inputs", data=inputs)
            group.create_dataset("targets", data=targets)
            group.create_dataset("gt_targets", data=gt_targets)

    def load(self, h5file):
        if type(h5file) is not h5py.File:
            raise ValueError("Dataset file format must be HDF5.")

        data = {}
        data["networks"] = {}
        data["inputs"] = {}
        data["targets"] = {}
        data["gt_targets"] = {}

        for i, k in enumerate(h5file.keys()):
            group = h5file[k]
            num_nodes = group["inputs"][...].shape[0]
            g = nx.empty_graph(num_nodes)
            g.add_edges_from(group["edge_list"][...].T)
            data["networks"][i] = g
            data["inputs"][i] = group["inputs"][...]
            data["targets"][i] = group["targets"][...]
            data["gt_targets"][i] = group["gt_targets"][...]

        self.data = data


class DegreeWeightedDataset(Dataset):
    def _get_weights_(self):
        weights = {}
        counts = {}
        degrees = []
        for i, g in self.networks.items():
            weights[i] = np.zeros(self.inputs[i].shape)
            degrees.append(list(dict(g.degree()).values()))
            for k in degrees[-1]:
                if k in counts:
                    counts[k] += 1
                else:
                    counts[k] = 1
        for i in self.networks.keys():
            for j, k in enumerate(degrees[i]):
                weights[i][:, j] = counts[k]
        return weights


class StateWeightedDataset(Dataset):
    def _get_weights_(self):
        weights = {}
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
