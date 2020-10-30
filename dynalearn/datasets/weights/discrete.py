import networkx as nx
import numpy as np

from .weight import Weight
from dynalearn.utilities import from_nary


class DiscreteStateWeight(Weight):
    def __init__(self, name="weights", max_window_size=3, bias=1.0):
        self.max_window_size = max_window_size
        Weight.__init__(self, name=name, max_num_samples=max_window_size, bias=bias)

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
        adj = network.to_array()
        degree = network.degree()
        for i, x in enumerate(states):
            s, ns = self._get_compound_states_(adj, x)
            for j in range(s.shape[0]):
                key = (s[j], degree[j])
                self._add_features_(("state", key))
            if pb is not None:
                pb.update()

    def _get_weights_(self, network, states, pb=None):
        weights = np.zeros((states.shape[0], states.shape[1]))
        z = sum(self.features.values())
        adj = network.to_array()
        degree = network.degree()
        for i, x in enumerate(states):
            s, ns = self._get_compound_states_(adj, x)
            for j in range(s.shape[0]):
                key = (s[j], degree[j])
                key = (s[j], k)
                weights[i, j] = self.features[("state", key)] / z
            if pb is not None:
                pb.update()
        return weights


class DiscreteCompoundStateWeight(DiscreteStateWeight):
    def _get_features_(self, network, states, pb=None):
        adj = network.to_array()
        for i, x in enumerate(states):
            s, ns = self._get_compound_states_(adj, x)
            for j in range(s.shape[0]):
                key = (s[j], *ns[j])
                self._add_features_(("state", key))
            if pb is not None:
                pb.update()

    def _get_weights_(self, network, states, pb=None):
        weights = np.zeros((states.shape[0], states.shape[1]))
        z = sum(self.features.values())
        adj = network.to_array()
        for i, x in enumerate(states):
            s, ns = self._get_compound_states_(adj, x)
            for j in range(s.shape[0]):
                key = (s[j], *ns[j])
                weights[i, j] = self.features[("state", key)] / z
            if pb is not None:
                pb.update()
        return weights
