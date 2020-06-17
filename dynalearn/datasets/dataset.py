import copy
import h5py
import networkx as nx
import numpy as np
import tqdm

from abc import abstractmethod
from itertools import islice, chain
from .sampler import Sampler
from dynalearn.utilities import to_edge_index
from dynalearn.config import Config


class Dataset(object):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        self.config = config
        self.sampler = Sampler(self)
        self.bias = config.bias
        if data is not None:
            self.data = data
        else:
            self._data = None
            self._weights = {}
            self.num_states = None
            self.indices = {}

    @abstractmethod
    def _generate_sequence_(self, m_network, m_dynamics, details, pb=None):
        raise NotImplemented()

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplemented()

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

        if experiment.verbose != 0 and experiment.verbose != 1:
            print("Generating training set")

        if experiment.verbose == 1:
            pb = tqdm.tqdm(
                range(details.num_networks * details.num_samples),
                "Generating training set",
            )
        else:
            pb = None

        self.data = self._generate_sequence(m_network, m_dynamics, details, pb=pb)
        if experiment.verbose == 1:
            pb.close()

    def partition(self, node_fraction, bias=0, pb=None):
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
                if pb is not None:
                    pb.update()
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
            num_nodes = group["inputs"][...].shape[-1]
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
