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


class WeightData(DataCollection):
    def __init__(self, name="weight_collection", max_num_samples=1000):
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
            weights = StateData(
                data=self._get_weights_(g, dataset.inputs[i].data, pb=pb)
            )
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
            elif (
                len(self.features[key]) > self.max_num_samples
                and self.max_num_samples != -1
            ):
                pass
            else:
                if isinstance(value, list):
                    self.features[key].extend(value)
                else:
                    self.features[key].append(value)

    def _get_kde_(self, samples):
        assert isinstance(samples, list)
        prob = {}
        x = np.array(samples)
        x = x.reshape(x.shape[0], -1).T
        mean = np.expand_dims(x.mean(axis=-1), -1)
        std = np.expand_dims(x.std(axis=-1), -1)
        x = (x - mean) / std
        return gaussian_kde(x), mean, std

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
        strength = get_node_strength(network)
        for k, s in zip(degree, strength):
            self._add_features_(("degree", k))
            self._add_features_(("strength", k), s)
            if pb is not None:
                pb.update()

    def _get_weights_(self, network, states, pb=None):
        degree = list(dict(network.degree()).values())
        weights = np.zeros((states.shape[0], states.shape[1]))

        z = 0
        for k, v in self.features:
            if k[0] == "degree":
                z += v
        for i, k in enumerate(degree):
            p = self._get_kde_probability_(self.features[("strength", k)])
            weights[:, i] = p * self.features[("degree", k)] / z
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
        s = np.array(
            [
                from_nary(ss[:, -self.window_size :], axis=-1, base=self.num_states)
                for ss in state
            ]
        )
        ns = np.zeros((state.shape[0], eff_num_states))
        for j in range(eff_num_states):
            ns[:, j] = adj @ (s == j)
        return s, ns

    def _get_features_(self, network, states, pb=None):
        eff_num_states = self.num_states ** self.window_size
        adj = nx.to_numpy_array(network)
        for i, x in states:
            s, ns = self._get_compound_states_(adj, x)
            for j in range(s.shape[0]):
                key = (s[j], *ns[j])
                self._add_features_(key)
            if pb is not None:
                pb.update()

    def _get_weights_(self, network, states, pb=None):
        weights = np.zeros((states.shape[0], states.shape[1]))
        z = sum(self.features.values())
        for i, x in states:
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
        degree = list(dict(network.degree()).values())
        if states.ndim == 3:
            states = states[:, :, -1]
        for i, s in enumerate(states):
            for j, (ss, k) in enumerate(zip(s, degree)):
                compound_state = [
                    np.concatenate([ss.reshape(-1), s[l].reshape(-1)])
                    for l in network.neighbors(j)
                ]
                self._add_features_(("degree", k))
                self._add_features_(("state-pair", k), compound_state)
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
                kde[k[1]], mean[k[1]], std[k[1]] = self._get_kde_(v)
        for i, s in enumerate(states):
            for j, (ss, k) in enumerate(zip(s, degree)):
                if k > 0:
                    compound_state = []
                    for l in network.neighbors(j):
                        compound_state.append(np.concatenate([ss, s[l]]))
                    compound_state = (
                        np.array(compound_state).reshape(len(compound_state), -1).T
                    )
                    cs = (compound_state - mean[k]) / std[k]
                    p = np.prod(kde[k](cs)) ** (1.0 / k)
                    weights[i, j] = self.features[("degree", k)] / z * p
                else:
                    weights[i, j] = self.features[("degree", k)] / z
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

                compound_state = []
                for l in network.neighbors(j):
                    ew = network.edges[j, l]["weight"]
                    compound_state.append(np.concatenate([ss, s[l], ew]))
                self._add_features_(("state-pair", k), compound_state)
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
                kde[k[1]], mean[k[1]], std[k[1]] = self._get_kde_(v)
        for i, s in enumerate(states):
            for j, (ss, k) in enumerate(zip(s, degree)):
                if k > 0:
                    compound_state = []
                    for l in network.neighbors(j):
                        ew = network.edges[j, l]["weight"]
                        compound_state.append(np.concatenate([ss, s[l], ew]))
                    compound_state = (
                        np.array(compound_state).reshape(len(compound_state), -1).T
                    )
                    cs = (compound_state - mean[k]) / std[k]
                    p = np.prod(kde[k](cs)) ** (1.0 / k)

                    weights[i, j] = self.features[("degree", k)] / z * p
                else:
                    weights[i, j] = self.features[("degree", k)] / z
            if pb is not None:
                pb.update()
        return weights
