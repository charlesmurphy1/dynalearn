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

    def _reduce_global_(self, states, network):
        return

    def _get_features_(self, network, states, pb=None):
        for i, s in enumerate(states):
            y = self._reduce_global_(s, network)
            if y is not None:
                self._add_features_("global", y)
            for j, ss in enumerate(s):
                k = network.degree(j)
                self._add_features_(("degree", int(k)))
                x = self._reduce_(j, s, network)
                if x is not None:
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
                kde[k] = KernelDensityEstimator(
                    samples=v, max_num_samples=self.max_num_samples
                )
            elif k == "global":
                kde[k] = KernelDensityEstimator(
                    samples=v, max_num_samples=self.max_num_samples
                )
        for i, s in enumerate(states):
            y = self._reduce_global_(s, network)
            if y is not None:
                p_g = kde["global"].pdf(y)
            else:
                p_g = 1.0
            assert p_g > 0, f"Encountered invalid value."
            for j, ss in enumerate(s):
                k = network.degree(j)
                x = self._reduce_(j, s, network)
                if x is not None:
                    p_s = gmean(kde[("state", k)].pdf(x))
                else:
                    p_s = 1.0
                assert p_s > 0, "Encountered invalid value."
                weights[i, j] = self.features[("degree", k)] / z * p_s * p_g
            if pb is not None:
                pb.update()
        return weights


class ContinuousGlobalStateWeight(ContinuousStateWeight):
    def _reduce_global_(self, states, network):
        return states.sum(0).reshape(-1)

    def _reduce_(self, index, states, network):
        return


class StrengthContinuousGlobalStateWeight(ContinuousStateWeight):
    def _reduce_global_(self, states, network):
        return states.sum(0)

    def _reduce_(self, index, states, network):
        s = np.array([0.0])
        for l in network.neighbors(index):
            if "weight" in network.edges[index, l]:
                s += network.edges[index, l]["weight"]
            else:
                s += np.array([1.0])
        return s.reshape(-1)


class StrengthContinuousStateWeight(ContinuousStateWeight):
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
