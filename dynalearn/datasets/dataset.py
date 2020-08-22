import copy
import h5py
import networkx as nx
import numpy as np
import tqdm

from abc import abstractmethod
from itertools import islice, chain
from .sampler import Sampler
from scipy.stats import gaussian_kde

from dynalearn.config import Config
from dynalearn.datasets import TransformList
from dynalearn.datasets.data import (
    DataCollection,
    NetworkData,
    StateData,
)
from dynalearn.datasets.transforms.getter import get as get_transforms
from dynalearn.utilities import to_edge_index, get_node_strength, collapse_networks


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
        self._state_weights = None
        self._network_weights = None
        self._indices = None
        self._rev_indices = None

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplemented()

    def __len__(self):
        return np.sum([s.size for s in self.data["inputs"].data_list])

    def __iter__(self):
        return self

    def __next__(self):
        return self[self.rev_indices[self.sampler()]]

    def generate(self, experiment):
        details = self.setup(experiment)
        self.transforms.setup(experiment)

        if self.verbose != 0 and self.verbose != 1:
            print("Generating training set")

        if self.verbose == 1:
            pb = tqdm.tqdm(
                range(details.num_networks * details.num_samples),
                "Generating training set",
            )
        else:
            pb = None

        self.data = self._generate_data_(details, pb=pb)
        if self.use_groundtruth:
            self._data["ground_truth"] = self._generate_groundtruth_(self._data)
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
                p = self.weights[i][j, index] ** (-bias)
                p /= p.sum()
                remove_nodes = np.random.choice(index, p=p, size=n, replace=False)
                weights[i][j] *= 0
                weights[i][j][remove_nodes] = self.weights[i][j][remove_nodes] * 1
                self.weights[i][j][remove_nodes] = 0
                if pb is not None:
                    pb.update()
        dataset.weights = weights
        dataset.indices = self.indices
        dataset.window_size = self.window_size
        dataset.window_step = self.window_step
        dataset.threshold_window_size = self.threshold_window_size
        dataset.num_states = self.num_states
        dataset.sampler.reset()
        return dataset

    def setup(self, experiment):
        self.m_networks = experiment.networks
        self.m_dynamics = experiment.dynamics
        self.window_size = experiment.model.window_size
        self.window_step = experiment.model.window_step
        self.threshold_window_size = experiment.train_details.threshold_window_size
        self.num_states = experiment.model.num_states
        self.verbose = experiment.verbose
        return experiment.train_details

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
        assert isinstance(h5file, h5py.Group)

        name = name or "data"
        if name in h5file:
            del h5file[name]
        group = h5file.create_group(name)
        self._save_data_(self._data, group)

        if len(self._transformed_data) > 0:
            name = f"transformed_{name}"
            if name in h5file:
                del h5file[name]
            group = h5file.create_group(name)
            self._save_data_(self._transformed_data, h5file[name])

    def load(self, h5file):
        assert isinstance(h5file, h5py.Group)

        self._data = {}
        self._transformed_data = {}

        if "data" in h5file:
            self._data = self._load_data_(h5file["data"])

        if self.use_transformed:
            if "transformed_data" in h5file:
                self._transformed_data = self._load_data_("transformed_data", h5file)
            elif len(self._data) > 0:
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
        self.weights = self._get_weights_(data)
        self.indices = self._get_indices_(data)
        self.sampler.reset()

    @property
    def inputs(self):
        return self.data["inputs"]

    @property
    def targets(self):
        if self.use_groundtruth and "ground_truth" in self._data:
            return self._data["ground_truth"]
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
        self._state_weights = {i: w.sum(-1) for i, w in self._weights.items()}
        self._network_weights = np.array(
            [w.sum(-1).sum(-1) for w in self._weights.values()]
        )

    @property
    def indices(self):
        return self._indices

    @indices.setter
    def indices(self, indices):
        self._indices = indices
        self._rev_indices = {(i, j): k for k, (i, j) in self._indices.items()}

    @property
    def rev_indices(self):
        return self._rev_indices

    @property
    def state_weights(self):
        return self._state_weights

    @property
    def network_weights(self):
        return self._network_weights

    def _generate_data_(self, details, pb=None):
        networks = DataCollection(name="networks")
        inputs = DataCollection(name="inputs")
        targets = DataCollection(name="targets")
        back_step = (self.window_size - 1) * self.window_step

        for i in range(details.num_networks):
            self.m_dynamics.network = self.m_networks.generate()

            networks.add(NetworkData(data=self.m_dynamics.network))

            x = self.m_dynamics.initial_state()
            in_data = np.zeros((*x.shape, self.window_size))
            inputs_data = np.zeros((details.num_samples, *in_data.shape))
            targets_data = np.zeros((details.num_samples, *x.shape))
            t = 0
            j = 0
            k = 0
            while j < details.num_samples:
                if t % self.window_step == 0:
                    in_data.T[k] = 1 * x.T
                    k += 1
                    if k == self.window_size:
                        inputs_data[j] = 1 * in_data
                        for _ in range(self.window_step):
                            y = self.m_dynamics.sample(x)
                            x = 1 * y
                        targets_data[j] = 1 * y
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
            inputs.add(StateData(data=inputs_data))
            targets.add(StateData(data=targets_data))
        data = {
            "networks": networks,
            "inputs": inputs,
            "targets": targets,
        }

        return data

    def _generate_groundtruth_(self, data):
        ground_truth = DataCollection(name="ground_truth")
        for i in range(data["networks"].size):
            g = data["networks"][i].data
            self.m_dynamics.network = g
            num_samples = data["inputs"][i].size
            gt_data = None

            for j, x in enumerate(data["inputs"][i].data):
                y = self.m_dynamics.predict(x)
                if gt_data is None:
                    gt_data = np.zeros((num_samples, *y.shape))
                gt_data[j] = y
            ground_truth.add(StateData(data=gt_data))
        return ground_truth

    def _transform_data_(self, data):
        return {k: v.copy().transform(self.transforms) for k, v in data.items()}

    def _get_indices_(self, data):
        if data["inputs"] is None or data["networks"] is None:
            return {}
        index = 0
        indices_dict = {}
        for i in range(data["networks"].size):
            for j in range(data["inputs"][i].size):
                indices_dict[index] = (i, j)
                index += 1
        return indices_dict

    def _get_weights_(self, data):
        return {
            i: np.ones((data["inputs"][i].size, g.number_of_nodes()))
            for i, g in enumerate(data["networks"].data)
        }

    def _save_data_(self, data, h5file):
        data["networks"].save(h5file)
        data["inputs"].save(h5file)
        data["targets"].save(h5file)
        if "ground_truth" in data:
            data["ground_truth"].save(h5file)

    def _load_data_(self, h5file):
        data = {
            "networks": DataCollection(name="networks"),
            "inputs": DataCollection(name="inputs"),
            "targets": DataCollection(name="targets"),
            "ground_truth": DataCollection(name="ground_truth"),
        }

        for d_type in ["networks", "inputs", "targets", "ground_truth"]:
            if d_type in h5file:
                group = h5file[d_type]
                for k, v in group.items():
                    if d_type == "networks":
                        d = NetworkData()
                    else:
                        d = StateData()
                    d.load(v)
                    data[d_type].add(d)
        return data


class DegreeWeightedDataset(Dataset):
    def _get_weights_(self, data):
        weights = {}
        counts = {}
        degrees = []
        for i in range(data["networks"].size):
            g = data["networks"][i].data
            n = g.number_of_nodes()
            weights[i] = np.zeros((data["inputs"][i].size, n))
            degrees.append(list(dict(g.degree()).values()))
            for k in degrees:
                if k in counts:
                    counts[k] += 1
                else:
                    counts[k] = 1
        for i in range(data["networks"].size):
            for j, k in enumerate(degrees[i]):
                weights[i][:, j] = counts[k]
        return weights


class StrengthWeightedDataset(Dataset):
    def _get_distribution_(self, data):
        degrees = []
        samples = {}
        counts = {}
        z = 0
        for i in range(data["networks"].size):
            g = data["networks"][i].data
            if isinstance(g, dict):
                g = collapse_networks(g)
            s = get_node_strength(g)["weight"]
            degrees = list(dict(g.degree()).values())
            z += g.number_of_nodes()
            for j, k in enumerate(degrees):
                if k in counts:
                    counts[k] += 1
                    samples[k].append(s[j])
                else:
                    counts[k] = 1
                    samples[k] = [s[j]]

        kde_dict = {}
        mean_dict = {}
        std_dict = {}
        p_k = {}
        for k, c in counts.items():
            x = np.array(samples[k])
            mean = np.mean(x)
            std = np.std(x)
            if std > 0:
                x = (x - mean) / std
                kde_dict[k] = gaussian_kde(x)
            else:
                kde_dict[k] = lambda x: 1.0
            mean_dict[k] = mean
            std_dict[k] = std
            p_k[k] = c / z
        return p_k, kde_dict, mean_dict, std_dict

    def _get_weights_(self, data):
        weights = {}
        p_k, kde, mean, std = self._get_distribution_(data)
        for i in range(data["networks"].size):
            g = data["networks"][i].data
            if isinstance(g, dict):
                g = collapse_networks(g)
            weights[i] = np.zeros((data["inputs"][i].size, g.number_of_nodes()))
            degrees = list(dict(g.degree()).values())

            s = get_node_strength(g)["weight"]
            for j, ss in enumerate(s):
                k = degrees[j]
                if std[k] == 0:
                    weights[i][:, j] = p_k[k]
                else:
                    ss = (ss - mean[k]) / std[k]
                    weights[i][:, j] = p_k[k] * kde[k](ss)
        return weights
