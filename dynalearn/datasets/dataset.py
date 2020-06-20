import copy
import h5py
import networkx as nx
import numpy as np
import tqdm

from abc import abstractmethod
from itertools import islice, chain
from .sampler import Sampler
from dynalearn.config import Config
from dynalearn.datasets import TransformList
from dynalearn.utilities import to_edge_index


class Dataset(object):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        self.config = config
        self.sampler = Sampler(self)
        self.bias = config.bias
        if "transforms" in config.__dict__:
            self.transforms = get_transforms(config.transforms)
        else:
            self.transforms = TransformList()
        self.use_transformed = len(self.transforms) > 0
        self.use_groundtruth = config.use_groundtruth

        self._data = {}
        self._transformed_data = {}
        self._weights = None
        self._indices = None
        self.rev_indices = None

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

    def generate(self, experiment):
        m_network = experiment.networks
        m_dynamics = experiment.dynamics
        details = experiment.train_details
        self.num_states = int(experiment.model.num_states)

        if experiment.verbose != 0 and experiment.verbose != 1:
            print("Generating training set")

        if experiment.verbose == 1:
            pb = tqdm.tqdm(
                range(details.num_networks * details.num_samples),
                "Generating training set",
            )
        else:
            pb = None

        self.data = self._generate_data_(m_network, m_dynamics, details, pb=pb)

        if self.use_groundtruth:
            self._data["gt_targets"] = self._generate_groundtruth_(
                self._data, m_dynamics
            )

        if experiment.verbose == 1:
            pb.close()

    def partition(self, node_fraction, bias=0, pb=None):
        dataset = type(self)(self.config)
        dataset._data = self._data
        if self.use_transformed:
            dataset._transformed_data = self._transformed_data
        weights = {i: np.zeros(w.shape) for i, w in self.weights.items()}
        for i in self.networks.keys():
            for j in range(self.inputs[i].shape[0]):
                index = np.where(self.weights[i][j] > 0)[0]
                n = np.random.binomial(index.shape[0], node_fraction)
                p = self.weights[i][j] ** (-bias)
                p /= p.sum()
                remove_nodes = np.random.choice(index, p=p, size=n, replace=False)

                weights[i][j] *= 0
                weights[i][j][remove_nodes] = self._weights[i][j][remove_nodes] * 1
                self._weights[i][j][remove_nodes] = 0
                if pb is not None:
                    pb.update()
        dataset.weights = weights
        dataset.indices = self.indices
        return dataset

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

        self._save_data_(self._data, h5file, name)
        if len(self._transformed_data) > 0:
            self._save_data_(self._transformed_data, h5file, f"transformed_{name}")

    def load(self, h5file):
        if type(h5file) is not h5py.File:
            raise ValueError("Dataset file format must be HDF5.")

        self._data = self._load_data_("data", h5file)
        if self.use_transformed:
            self._transformed_data = self._load_data_("transformed_data", h5file)
            if len(self._transformed_data) == 0:
                self._transformed_data = self._transform_data_(self._data)

    @property
    def data(self):
        if self.use_transformed:
            return self._transformed_data
        else:
            return self._data

    @data.setter
    def data(self, data):
        self._data = data
        if self.use_transformed:
            self._transformed_data = self._transform_data_(data)
        self.weights = self._get_weights_()
        self.indices = self._get_indices_()
        self.sampler.reset()

    @property
    def inputs(self):
        return self.data["inputs"]

    @property
    def targets(self):
        if self.use_groundtruth:
            return self.data["gt_targets"]
        else:
            return self.data["targets"]

    @property
    def networks(self):
        return self.data["networks"]

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

    def _generate_data_(self, m_network, m_dynamics, details, pb=None):
        networks = {}
        inputs = {}
        targets = {}
        gt_targets = {}

        for i in range(details.num_networks):
            m_dynamics.network = m_network.generate()
            x = m_dynamics.initial_state()
            g = m_dynamics.network

            networks[i] = g
            inputs[i] = np.zeros((details.num_samples, m_network.num_nodes))
            targets[i] = np.zeros((details.num_samples, m_network.num_nodes))

            for j in range(details.num_samples):
                y = m_dynamics.sample(x)
                inputs[i][j] = x
                targets[i][j] = y
                if j % details.resampling_time == 0:
                    x = m_dynamics.initial_state()
                else:
                    x = y
                if pb is not None:
                    pb.update()
        data = {
            "networks": networks,
            "inputs": inputs,
            "targets": targets,
        }

        return data

    def _generate_groundtruth_(self, data, m_dynamics):
        ground_truth = {}

        for i, g in data["networks"].items():
            m_dynamics.network = g
            num_samples = data["inputs"][i].shape[0]
            num_nodes = data["inputs"][i].shape[1]
            ground_truth[i] = np.zeros((num_samples, num_nodes, self.num_states))
            for j, x in enumerate(data["inputs"][i]):
                ground_truth[i][j] = m_dynamics.predict(x)
        return ground_truth

    def _transform_data_(self, data):
        _data = data.copy()
        for i in _data["networks"]:
            _data["networks"][i] = self.transforms(_data["networks"][i])
            for j in range(_data["inputs"][i].shape[0]):
                _data["inputs"][i][j] = self.transforms(_data["inputs"][i][j])
                _data["targets"][i][j] = self.transforms(_data["targets"][i][j])
        return _data

    def _get_indices_(self):
        if self.data["inputs"] is None or self.data["networks"] is None:
            return {}
        index = 0
        indices_dict = {}
        for i in self.data["networks"].keys():
            for j in range(self.data["inputs"][i].shape[0]):
                indices_dict[index] = (i, j)
                index += 1
        return indices_dict

    def _get_weights_(self):
        return {
            i: np.ones(self.data["inputs"][i].shape)
            for i, g in self.data["networks"].items()
        }

    def _save_data_(self, data, h5file, name):
        for i, g in data["networks"].items():
            if len(g.edges()) > 0:
                edge_list = to_edge_index(g)
            else:
                edge_list = np.zeros((0, 2)).astype("int")

            if f"{name}{i}" in h5file:
                del h5file[f"{name}{i}"]
            group = h5file.create_group(f"{name}{i}")
            group.create_dataset("edge_list", data=edge_list)
            group.create_dataset("inputs", data=data["inputs"][i])
            group.create_dataset("targets", data=data["targets"][i])

    def _load_data_(self, name, h5file):
        data = {}
        data["networks"] = {}
        data["inputs"] = {}
        data["targets"] = {}

        for i, k in enumerate(h5file.keys()):
            group = h5file[k]
            if k[: len(name)] == name:
                num_nodes = group["inputs"][...].shape[-1]
                g = nx.empty_graph(num_nodes)
                g.add_edges_from(group["edge_list"][...].T)
                data["networks"][i] = g
                data["inputs"][i] = group["inputs"][...]
                data["targets"][i] = group["targets"][...]

        return data


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
