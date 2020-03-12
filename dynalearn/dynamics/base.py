"""

dynamical_network.py

Created by Charles Murphy on 26-06-18.
Copyright © 2018 Charles Murphy. All rights reserved.
Quebec, Canada

Defines the class DynamicalNetwork which generate network on which a dynamical
process occurs.

"""

import networkx as nx
import numpy as np
import pickle
import torch as pt
import os
from abc import ABC, abstractmethod
from math import ceil


class Dynamics(ABC):
    """
		Base class for dynamical network.

		**Parameters**
		graph : nx.Graph
			A graph on which the dynamical process occurs.

		filename : String : (default = ``None``)
			Name of file for saving activity states. If ``None``, it does not save the states.

		full_data_mode : Bool : (default = ``False``)


	"""

    def __init__(self, num_states):
        """
		Initializes a Dynamics object.

		"""
        self._num_states = num_states
        self._graph = None
        self._adj = None
        self._num_nodes = None
        self._degree = None

    @abstractmethod
    def initial_states(self):
        """
		Initializes the nodes activity states. (virtual) (private)

		"""
        raise NotImplementedError("self.initial_states() has not been impletemented")

    @abstractmethod
    def predict(self, states):
        """
		Computes the next activity states probability distribution. (virtual) (private)

		"""
        raise NotImplementedError("self.transition() has not been impletemented")

    @abstractmethod
    def sample(self, states):
        """
		Computes the next activity states. (virtual) (private)

		"""
        raise NotImplementedError("sample has not been impletemented")

    @abstractmethod
    def is_dead(self, states):
        """
		Computes the next activity states. (virtual) (private)

		"""
        raise NotImplementedError("sample has not been impletemented")

    @property
    def graph(self):
        if self._graph is None:
            raise ValueError("No graph has been parsed to the dynamics.")
        else:
            return self._graph

    @graph.setter
    def graph(self, graph):
        self._graph = graph
        self._adj = nx.to_numpy_array(graph)
        self._num_nodes = self._graph.number_of_nodes()
        self._degree = np.sum(self._adj, axis=1)

    @property
    def adj(self):
        if self._adj is None:
            raise ValueError("No graph has been parsed to the dynamics.")
        else:
            return self._adj

    @adj.setter
    def adj(self, adj):
        self._adj = adj
        self.graph = nx.from_numpy_array(adj)
        self._num_nodes = self._graph.number_of_nodes()
        self._degree = np.sum(self._adj, axis=1)

    @property
    def num_nodes(self):
        if self._num_nodes is None:
            raise ValueError("No graph has been parsed to the dynamics.")
        else:
            return self._num_nodes

    @property
    def degree(self):
        if self._degree is None:
            raise ValueError("No graph has been parsed to the dynamics.")
        else:
            return self._degree

    @property
    def num_states(self):
        return self._num_states


class Epidemics(Dynamics):
    def __init__(self, params, state_label):
        super(Epidemics, self).__init__(len(state_label))
        self.params = params
        self.state_label = state_label
        self.inv_state_label = {state_label[i]: i for i in state_label}

    def sample(self, states):
        p = self.predict(states)
        dist = pt.distributions.Categorical(pt.tensor(p))
        return np.array(dist.sample())

    def state_degree(self, states):
        if len(states.shape) < 2:
            states = states.reshape(1, self.num_nodes)

        state_l = {
            s: np.matmul(states == self.state_label[s], self.adj).squeeze()
            for s in self.state_label
        }

        return state_l


class SingleEpidemics(Epidemics):
    def __init__(self, params, state_label):

        if "S" not in state_label or "I" not in state_label:
            raise ValueError("state_label must contain states 'S' and 'I'.")
        super(SingleEpidemics, self).__init__(params, state_label)

    def initial_states(self):
        if self.params["init"] is None:
            p = self.params["init"]
        else:
            p = np.random.rand()
        n_infected = np.random.binom(self.num_nodes, p)
        nodeset = np.array(list(self.graph.nodes()))
        ind = np.random.choice(nodeset, size=init_n_infected, replace=False)
        states = np.ones(self.num_nodes) * self.state_label["S"]
        states[ind] = self.state_label["I"]

        return states

    def is_dead(self, states):
        if np.all(states == self.state_label["S"]):
            return True
        else:
            return False


class DoubleEpidemics(Epidemics):
    def __init__(self, params, state_label):
        if (
            "SS" not in state_label
            or "SI" not in state_label
            or "IS" not in state_label
            or "II" not in state_label
        ):
            raise ValueError("state_label must contain states 'S' and 'I'.")
        super(DoubleEpidemics, self).__init__(params, state_label)

    def initial_states(self):
        if self.params["init"] is None:
            p1 = 1 - np.sqrt(1 - self.params["init"])
            p2 = 1 - np.sqrt(1 - self.params["init"])
        else:
            p1 = np.random.rand()
            p2 = np.random.rand()
        n1_infected = np.random.binom(self.num_nodes, p1)
        n2_infected = np.random.binom(self.num_nodes, p2)
        nodeset = np.array(list(self.graph.nodes()))
        ind1 = np.random.choice(nodeset, size=n1_infected, replace=False)
        ind2 = np.random.choice(nodeset, size=n2_infected, replace=False)
        ind3 = np.intersect1d(ind1, ind2)
        states = np.ones(self.num_nodes) * self.state_label["SS"]
        states[ind1] = self.state_label["IS"]
        states[ind2] = self.state_label["SI"]
        states[ind3] = self.state_label["II"]

        return states

    def is_dead(self, states):
        if np.all(states == self.state_label["SS"]):
            return True
        else:
            return False
