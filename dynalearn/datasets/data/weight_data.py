import networkx as nx
import numpy as np
import tqdm

from abc import abstractmethod
from scipy.stats.kde import gaussian_kde
from .data import DataCollection
from .state_data import StateData
from dynalearn.utilities import (
    collapse_networks,
    get_node_strength,
    get_edge_weights,
    from_nary,
)


class KernelDensityEstimator:
    def __init__(self, samples, max_num_samples=1000):
        assert isinstance(samples, list)
        self.samples = samples
        if isinstance(samples[0], np.ndarray):
            self.shape = samples[0].shape
        elif isinstance(samples[0], (int, float)):
            self.shape = (1,)
        for s in samples:
            if isinstance(s, (int, float)):
                s = np.array([s])
            assert s.shape == self.shape
        self.max_num_samples = max_num_samples
        self.kde = None
        self._mean = None
        self._std = None
        self._norm = None
        self._index = None
        self.get_kde()

    def pdf(self, x):
        if isinstance(x, list):
            x = np.array(x)
            x = x.reshape(x.shape[0], -1).T
        if x.shape == self.shape:
            x = np.expand_dims(x, -1)
        assert x.shape[:-1] == self.shape

        if self.kde is None:
            return np.ones(x.shape[-1]) / self._norm
        else:
            y = (x[self._index] - self._mean[self._index]) / self._std[self._index]
            p = self.kde.pdf(y) / self._norm
            return p

    def get_kde(self):
        if len(self.samples) <= 1:
            self._norm = 1
            return
        x = np.array(self.samples)
        x = x.reshape(x.shape[0], -1).T
        mean = np.expand_dims(x.mean(axis=-1), -1)
        std = np.expand_dims(x.std(axis=-1), -1)
        if np.all(std < 1e-8):
            self._norm = len(self.samples)
            return
        self._index = np.where(std > 1e-8)[0]
        y = (x[self._index] - mean[self._index]) / std[self._index]
        if self.max_num_samples < y.shape[-1]:
            ind = np.random.choice(range(y.shape), size=self.max_num_samples)
            y = y[:, ind]
        self.kde = gaussian_kde(y)
        self._mean = mean
        self._std = std
        self._norm = self.kde.pdf(y).sum()
        self.samples = []


class WeightData(DataCollection):
    def __init__(self, name="weight_collection", max_num_samples=10000):
        DataCollection.__init__(self, name=name)
        self.max_num_samples = max_num_samples
        self.features = {}

    def _get_features_(self, network, states, pb=None):
        if pb is not None:
            pb.update()
        return

    def _get_weights_(self, network, states, pb=None):
        if pb is not None:
            pb.update()
        return np.ones((state.shape[0], state.shape[1]))

    def compute(self, dataset, verbose=0):
        self.setUp(dataset)
        if verbose != 0 and verbose != 1:
            print("Computing weights")

        if verbose == 1:
            pb = tqdm.tqdm(range(self.num_updates), "Computing weights",)
        else:
            pb = None
        self.compute_features(dataset, pb=pb)
        self.compute_weights(dataset, pb=pb)
        self.clear()
        if verbose == 1:
            pb.close()

    def setUp(self, dataset):
        self.num_updates = 2 * dataset.networks.size

    def compute_features(self, dataset, pb=None):

        for i in range(dataset.networks.size):
            g = dataset.networks[i].data
            if isinstance(g, dict):
                g = collapse_networks(g)
            self._get_features_(g, dataset.inputs[i].data, pb=pb)
        return

    def compute_weights(self, dataset, pb=None):
        for i in range(dataset.networks.size):
            g = dataset.networks[i].data
            if isinstance(g, dict):
                g = collapse_networks(g)
            w = self._get_weights_(g, dataset.inputs[i].data, pb=pb)
            weights = StateData(data=w)
            self.add(weights)

    def _add_features_(self, key, value=None):
        if value is None:
            if key not in self.features:
                self.features[key] = 1
            else:
                self.features[key] += 1
        else:
            if key not in self.features:
                if isinstance(value, list):
                    self.features[key] = value
                else:
                    self.features[key] = [value]
            else:
                if isinstance(value, list):
                    self.features[key].extend(value)
                else:
                    self.features[key].append(value)

    def clear(self):
        self.features = {}

    def to_state_weights(self):
        state_weights = DataCollection()
        for i in range(self.size):
            state_weights.add(StateData(data=self.data_list[i].data.sum(-1)))
        return state_weights

    def to_network_weights(self):
        network_weights = StateData()
        w = []
        for i in range(self.size):
            w.append(self.data_list[i].data.sum())
        network_weights.data = np.array(w)
        return network_weights


class DegreeWeightData(WeightData):
    def __init__(self, name="weight_collection"):
        WeightData.__init__(self, name=name, max_num_samples=-1)

    def setUp(self, dataset):
        self.num_updates = 2 * np.sum(
            [
                dataset.networks[i].data.number_of_nodes()
                for i in range(dataset.networks.size)
            ]
        )

    def _get_features_(self, network, states, pb=None):
        degree = list(dict(network.degree()).values())
        for k in degree:
            self._add_features_(k)
            if pb is not None:
                pb.update()

    def _get_weights_(self, network, states, pb=None):
        degree = list(dict(network.degree()).values())
        weights = np.zeros((states.shape[0], states.shape[1]))

        z = sum(self.features.values())
        for i, k in enumerate(degree):
            weights[:, i] = self.features[k] / z
            if pb is not None:
                pb.update()
        return weights


class NodeStrengthWeightData(WeightData):
    def setUp(self, dataset):
        self.num_updates = 2 * np.sum(
            [
                dataset.networks[i].data.number_of_nodes()
                for i in range(dataset.networks.size)
            ]
        )

    def _get_features_(self, network, states, pb=None):
        degree = list(dict(network.degree()).values())
        for i, k in enumerate(degree):
            self._add_features_(("degree", k))
            for j in network.neighbors(i):
                if "weight" in network.edges[i, j]:
                    ew = network.edges[i, j]["weight"]
                else:
                    ew = 1
                self._add_features_(("weight", k), ew)
            if pb is not None:
                pb.update()

    def _get_weights_(self, network, states, pb=None):
        degree = list(dict(network.degree()).values())
        weights = np.zeros((states.shape[0], states.shape[1]))

        z = 0
        kde = {}
        mean = {}
        std = {}
        for k, v in self.features.items():
            if k[0] == "degree":
                z += v
            elif k[0] == "weight":
                kde[k[1]] = KernelDensityEstimator(v)
        for i, k in enumerate(degree):
            ew = []
            for j in network.neighbors(i):
                if "weight" in network.edges[i, j]:
                    ew.append(network.edges[i, j]["weight"])
                else:
                    ew.append(1)
            p = np.prod(kde[k].pdf(ew)) ** (1.0 / k)
            weights[:, i] = self.features[("degree", k)] / z * p

            if pb is not None:
                pb.update()
        return weights


class DiscreteStateWeightData(WeightData):
    def __init__(self, name="weight_collection", max_window_size=3):
        self.max_window_size = max_window_size
        WeightData.__init__(self, name=name, max_num_samples=-1)

    def setUp(self, dataset):
        self.num_states = dataset.num_states
        if dataset.window_size > self.max_window_size:
            self.window_size = self.max_window_size
        else:
            self.window_size = dataset.window_size
        self.num_updates = 2 * np.sum(
            [dataset.inputs[i].data.shape[0] for i in range(dataset.networks.size)]
        )

    def _get_compound_states_(self, adj, state):
        eff_num_states = self.num_states ** self.window_size
        s = np.array(
            [from_nary(ss[-self.window_size :], base=self.num_states) for ss in state]
        )
        ns = np.zeros((state.shape[0], eff_num_states))
        for j in range(eff_num_states):
            ns[:, j] = adj @ (s == j)
        return s, ns

    def _get_features_(self, network, states, pb=None):
        adj = nx.to_numpy_array(network)
        for i, x in enumerate(states):
            s, ns = self._get_compound_states_(adj, x)
            for j in range(s.shape[0]):
                key = (s[j], *ns[j])
                self._add_features_(key)
            if pb is not None:
                pb.update()

    def _get_weights_(self, network, states, pb=None):
        weights = np.zeros((states.shape[0], states.shape[1]))
        z = sum(self.features.values())
        adj = nx.to_numpy_array(network)
        for i, x in enumerate(states):
            s, ns = self._get_compound_states_(adj, x)
            for j in range(s.shape[0]):
                key = (s[j], *ns[j])
                weights[i, j] = self.features[key] / z
            if pb is not None:
                pb.update()
        return weights


class ContinuousStateWeightData(WeightData):
    def setUp(self, dataset):
        self.num_updates = 2 * np.sum(
            [dataset.inputs[i].data.shape[0] for i in range(dataset.networks.size)]
        )

    def _get_features_(self, network, states, pb=None):
        for i, s in enumerate(states):
            for j, ss in enumerate(s):
                cs = []
                k = network.degree(j)
                for l in network.neighbors(j):
                    cs.append(np.concatenate([ss.reshape(-1), s[l].reshape(-1)]))
                self._add_features_(("degree", int(k)))
                self._add_features_(("state-pair", int(k)), cs)
            if pb is not None:
                pb.update()

    def _get_weights_(self, network, states, pb=None):
        weights = np.zeros((states.shape[0], states.shape[1]))
        z = 0
        kde = {}
        pp = {}
        for k, v in self.features.items():
            if k[0] == "degree":
                z += v
            elif k[0] == "state-pair":
                x = np.array(v)
                kde[k[1]] = KernelDensityEstimator(
                    v, max_num_samples=self.max_num_samples
                )
                pp[k[1]] = kde[k[1]].pdf(v)
                assert np.all(pp[k[1]] > 0)
        for i, s in enumerate(states):
            for j, ss in enumerate(s):
                cs = []
                k = network.degree(j)
                for l in network.neighbors(j):
                    cs.append(np.concatenate([ss.reshape(-1), s[l].reshape(-1)]))
                p = np.prod(kde[k].pdf(cs)) ** (1.0 / k)
                weights[i, j] = self.features[("degree", k)] / z * p
            if pb is not None:
                pb.update()
        return weights


class NodeStrengthContinuousStateWeightData(WeightData):
    def setUp(self, dataset):
        self.num_updates = 2 * np.sum(
            [dataset.inputs[i].data.shape[0] for i in range(dataset.networks.size)]
        )

    def _get_features_(self, network, states, pb=None):
        degree = list(dict(network.degree()).values())
        if states.ndim == 3:
            states = states[:, :, -1]
        for i, s in enumerate(states):
            for j, (ss, k) in enumerate(zip(s, degree)):
                self._add_features_(("degree", k))

                cs = []
                for l in network.neighbors(j):
                    ew = np.array([network.edges[j, l]["weight"]])
                    cs.append(np.concatenate([ss.squeeze(), s[l].squeeze(), ew]))
                self._add_features_(("state-pair", k), cs)
            if pb is not None:
                pb.update()

    def _get_weights_(self, network, states, pb=None):
        weights = np.zeros((states.shape[0], states.shape[1]))
        degree = list(dict(network.degree()).values())
        z = 0
        kde = {}
        mean = {}
        std = {}
        for k, v in self.features.items():
            if k[0] == "degree":
                z += v
            elif k[0] == "state-pair":
                kde[k[1]] = KernelDensityEstimator(v)
        for i, s in enumerate(states):
            for j, (ss, k) in enumerate(zip(s, degree)):
                if k > 0:
                    cs = []
                    for l in network.neighbors(j):
                        ew = np.array([network.edges[j, l]["weight"]])
                        cs.append(np.concatenate([ss.squeeze(), s[l].squeeze(), ew]))
                    p = np.prod(kde[k].pdf(cs)) ** (1.0 / k)
                    weights[i, j] = self.features[("degree", k)] / z * p
                else:
                    weights[i, j] = self.features[("degree", k)] / z
            if pb is not None:
                pb.update()
        return weights
