import networkx as nx
import numpy as np

from scipy.stats import gmean
from .weight import Weight
from .kde import KernelDensityEstimator


class ContinuousStateWeight(Weight):
    def __init__(self, name="weight_collection", reduce=True):
        self.reduce = reduce
        Weight.__init__(self, name=name, max_num_samples=10000)

    def setUp(self, dataset):
        self.num_updates = 2 * np.sum(
            [dataset.inputs[i].data.shape[0] for i in range(dataset.networks.size)]
        )

    def _reduce_(self, index, states, network):
        x = states[index].reshape(-1)
        if self.reduce:
            x = np.array([x.sum()])
        return x

    def _get_features_(self, network, states, pb=None):
        for i, s in enumerate(states):
            for j, ss in enumerate(s):
                k = network.degree(j)
                x = self._reduce_(j, s, network)
                self._add_features_(("degree", int(k)))
                self._add_features_(("state", int(k)), x)
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
            elif k[0] == "state":
                kde[k[1]] = KernelDensityEstimator(
                    v, max_num_samples=self.max_num_samples
                )
        for i, s in enumerate(states):
            for j, ss in enumerate(s):
                x = self._reduce_(j, s, network)
                k = network.degree(j)
                p = gmean(kde[k].pdf(x))
                assert p > 0, "Encountered invalid value."
                weights[i, j] = self.features[("degree", k)] / z * p
            if pb is not None:
                pb.update()
        return weights


class StrengthContinuousStateWeight(Weight):
    def _reduce_(self, index, states, network):
        x = states[index].reshape(-1)
        if self.reduce:
            x = np.array([x.sum()])
        s = np.array([0.0])
        for l in network.neighbors(index):
            if "weight" in network.edges[index, l]:
                s += network.edges[index, l]["weight"]
            else:
                s += np.array([1.0])
        return np.concatenate([x, s])


class ContinuousCompoundStateWeight(ContinuousStateWeight):
    def _reduce_(self, index, states, network):
        x = []
        _x = states[index].reshape(-1)
        if self.reduce:
            _x = np.array([_x.sum()])
        for l in network.neighbors(index):
            _y = states[l].reshape(-1)
            if self.reduce:
                _y = np.array([_y.sum()])
            x.append(np.concatenate([_x, _y]))
        return x


class StrengthContinuousCompoundStateWeight(ContinuousStateWeight):
    def _reduce_(self, index, states, network):
        x = []
        s = states[index]
        for l in network.neighbors(index):
            _x = s.reshape(-1)
            _y = states[l].reshape(-1)
            if "weight" in network.edges[index, l]:
                _w = np.array([network.edges[index, l]["weight"]])
            else:
                _w = np.array([1.0])
            if self.reduce:
                _x = np.array([_x.sum()])
                _y = np.array([_y.sum()])
            x.append(np.concatenate([_x, _y, _w]))
        return x
