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
from dynalearn.datasets import Data, WindowedData, NetworkData
from dynalearn.datasets.transforms.getter import get as get_transforms
from dynalearn.utilities import to_edge_index


class Dataset(object):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        self.config = config
        self.bias = config.bias
        self.use_groundtruth = config.use_groundtruth

        self.sampler = Sampler(self)

        if "transforms" in config.__dict__:
            self.transforms = get_transforms(config.transforms)
        else:
            self.transforms = TransformList()
        self.use_transformed = len(self.transforms) > 0

        self._data = {}
        self._transformed_data = {}
        self._weights = None
        self._indices = None
        self._rev_indices = None
        self.m_dynamics = None
        self.m_networks = None
        self.window_size = None
        self.window_step = None

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplemented()

    def __len__(self):
        return np.sum([s.size for i, s in self.data["inputs"].items()])

    def __iter__(self):
        return self

    def __next__(self):
        return self[self._rev_indices[self.sampler()]]

    def generate(self, experiment):
        details = experiment.train_details
        self.m_networks = experiment.networks
        self.m_dynamics = experiment.dynamics
        self.window_size = experiment.model.window_size
        self.window_step = experiment.model.window_step
        self.threshold_window_size = experiment.train_details.threshold_window_size
        self.num_states = experiment.model.num_states
        self.transforms.setup(experiment)

        if experiment.verbose != 0 and experiment.verbose != 1:
            print("Generating training set")

        if experiment.verbose == 1:
            pb = tqdm.tqdm(
                range(details.num_networks * details.num_samples),
                "Generating training set",
            )
        else:
            pb = None

        self.data = self._generate_data_(details, pb=pb)
        if self.use_groundtruth:
            gt_data = self._generate_groundtruth_(self._data)
            self._data["gt_targets"] = {}
            for k, d in gt_data.items():
                self._data["gt_targets"][k] = Data(data=d, shape=d.shape[1:])

        if experiment.verbose == 1:
            pb.close()

    def partition(self, node_fraction, bias=0, pb=None):
        dataset = type(self)(self.config)
        dataset._data = self._data
        if self.use_transformed:
            dataset._transformed_data = self._transformed_data
        weights = {i: np.zeros(w.shape) for i, w in self.weights.items()}
        for i in range(self.networks.size):
            for j in range(self.inputs[i].size):
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
        dataset.window_size = self.window_size
        dataset.window_step = self.window_step
        dataset.threshold_window_size = self.threshold_window_size
        dataset.sampler.reset()
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
        self._data = {}
        self._transformed_data = {}
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
            return self._data["gt_targets"]
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
        self._rev_indices = {(i, j): k for k, (i, j) in self.indices.items()}

    def _generate_data_(self, details, pb=None):
        networks = NetworkData()
        inputs = {}
        targets = {}
        gt_targets = {}
        back_step = (self.window_size - 1) * self.window_step

        for i in range(details.num_networks):
            self.m_dynamics.network = self.m_networks.generate()

            networks.add(self.m_dynamics.network)

            x = self.m_dynamics.initial_state()
            in_data = np.zeros((self.window_size, *x.shape))

            inputs[i] = Data(name=f"data{i}", shape=in_data.shape)
            targets[i] = Data(name=f"data{i}", shape=x.shape)
            t = 0
            j = 0
            k = 0
            while j < details.num_samples:
                if t % self.window_step == 0:
                    in_data[k] = 1 * x
                    k += 1
                    if k == self.window_size:
                        inputs[i].add(in_data)
                        for _ in range(self.window_step):
                            y = self.m_dynamics.sample(x)
                            x = 1 * y
                        targets[i].add(y)
                        k = 0
                        j += 1
                        if pb is not None:
                            pb.update()
                        if j % details.resampling == 0 and details.resampling != -1:
                            x = self.m_dynamics.initial_state()
                    else:
                        x = self.m_dynamics.sample(x)
                else:
                    x = self.m_dynamics.sample(x)

                t += 1
        data = {
            "networks": networks,
            "inputs": inputs,
            "targets": targets,
        }

        return data

    def _generate_groundtruth_(self, data):
        ground_truth = {}

        for i, g in enumerate(data["networks"].data):
            self.m_dynamics.network = g
            num_samples = data["inputs"][i].size
            for j, x in enumerate(data["inputs"][i].data):
                y = self.m_dynamics.predict(x)
                if i not in ground_truth:
                    ground_truth[i] = np.zeros((num_samples, *y.shape))
                ground_truth[i][j] = y
        return ground_truth

    def _transform_data_(self, data):
        _data = {"networks": data["networks"].copy(), "inputs": {}, "targets": {}}
        _data["networks"].transform(self.transforms)
        for i in range(data["networks"].size):
            _data["inputs"][i] = data["inputs"][i].copy()
            _data["targets"][i] = data["targets"][i].copy()
            _data["inputs"][i].transform(self.transforms)
            _data["targets"][i].transform(self.transforms)
        return _data

    def _get_indices_(self):
        if self.data["inputs"] is None or self.data["networks"] is None:
            return {}
        index = 0
        indices_dict = {}
        for i in range(self.data["networks"].size):
            for j in range(self.data["inputs"][i].size):
                indices_dict[index] = (i, j)
                index += 1
        return indices_dict

    def _get_weights_(self):
        return {
            i: np.ones((self.data["inputs"][i].size, *self.data["inputs"][i].shape))
            for i, g in enumerate(self.data["networks"].data)
        }

    def _save_data_(self, data, h5file, name):
        for i, g in enumerate(data["networks"].data):
            if len(g.edges()) > 0:
                edge_list = to_edge_index(g)
            else:
                edge_list = np.zeros((0, 2)).astype("int")

            if f"{name}{i}" in h5file:
                del h5file[f"{name}{i}"]
            group = h5file.create_group(f"{name}{i}")
            group.create_dataset("edge_list", data=edge_list)
            group.create_dataset("inputs", data=data["inputs"][i].data)
            group.create_dataset("targets", data=data["targets"][i].data)

    def _load_data_(self, name, h5file):
        data = {}
        data["networks"] = NetworkData()
        data["inputs"] = {}
        data["targets"] = {}

        for i, k in enumerate(h5file.keys()):
            group = h5file[k]
            if k[: len(name)] == name:
                num_nodes = group["inputs"][...].shape[1]
                g = nx.empty_graph(num_nodes)
                g.add_edges_from(group["edge_list"][...].T)
                data["networks"].add(g)
                data["inputs"][i] = Data(name=f"data{i}", data=group["inputs"][...],)
                data["targets"][i] = Data(name=f"data{i}", data=group["targets"][...])

        return data


class DegreeWeightedDataset(Dataset):
    def _get_weights_(self):
        weights = {}
        counts = {}
        degrees = []
        for i in range(self.networks.size):
            g = self.networks.data[i]
            n = g.number_of_nodes()
            weights[i] = np.zeros((self.inputs[i].size, n))
            degrees.append(list(dict(g.degree()).values()))
            for k in degrees[-1]:
                if k in counts:
                    counts[k] += 1
                else:
                    counts[k] = 1
        for i in range(self.networks.size):
            for j, k in enumerate(degrees[i]):
                weights[i][:, j] = counts[k]
        return weights
