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
        self._degree = None

    @abstractmethod
    def initial_states(self):
        """
		Initializes the nodes activity states. (virtual) (private)

		"""
        raise NotImplementedError("self.initial_states() has not been impletemented")

    @abstractmethod
    def predict(self, states=None, ajd=None):
        """
		Computes the next activity states probability distribution. (virtual) (private)

		"""
        raise NotImplementedError("self.transition() has not been impletemented")

    @abstractmethod
    def sample(self, states=None, ajd=None):
        """
		Computes the next activity states. (virtual) (private)

		"""
        raise NotImplementedError("sample has not been impletemented")

    @abstractmethod
    def get_avg_state(self):
        """
		Get the average states. (virtual)

		**Returns**
		avg_state : Activity

		"""
        raise NotImplementedError("self.get_avg_states has not been implemented.")

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
        self._degree = np.sum(self._adj, axis=1)
        self.initial_states()

    @property
    def adj(self):
        if self._adj is None:
            raise ValueError("No graph has been parsed to the dynamics.")
        else:
            return self._adj

    @adj.setter
    def adj(self, adj):
        self._adj = adj
        self._graph = nx.from_numpy_array(adj)
        self._degree = np.sum(adj, axis=1)
        self.initial_states()

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

    def sample(self, states=None, ajd=None):
        p = self.predict(states=states, adj=ajd)
        dist = pt.distributions.Categorical(pt.tensor(p))
        return np.array(dist.sample())

    def state_degree(self, states):
        N = self.adj.shape[0]
        if len(states.shape) < 2:
            states = states.reshape(1, N)

        state_l = {
            s: np.matmul(states == self.state_label[s], self.adj).squeeze()
            for s in self.state_label
        }

        return state_l

    def get_avg_state(self):
        N = self.graph.number_of_nodes()
        state_dict = {l: np.zeros(N) for l in self.state_label}

        for v in self.graph.nodes():
            label = self.inv_state_label[self.states[v]]
            state_dict[label][v] = 1

        avg_states = {l: np.mean(state_dict[l]) for l in state_dict}
        std_states = {l: np.std(state_dict[l]) for l in state_dict}


class SingleEpidemics(Epidemics):
    def __init__(self, params, state_label):
        if "S" not in state_label or "I" not in state_label:
            raise ValueError("state_label must contain states 'S' and 'I'.")
        super(SingleEpidemics, self).__init__(params, state_label)

    def initial_states(self, graph=None):
        if graph is not None:
            self.graph = graph

        N = self.graph.number_of_nodes()
        if self.params["init"] is not None:
            init_n_infected = ceil(N * self.params["init"])
        else:
            init_n_infected = np.random.choice(range(N))
        nodeset = np.array(list(self.graph.nodes()))
        ind = np.random.choice(nodeset, size=init_n_infected, replace=False)
        states = np.ones(N) * self.state_label["S"]
        states[ind] = self.state_label["I"]

        self.continue_simu = True
        return states


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

    def initial_states(self, graph=None):
        if graph is not None:
            self.graph = graph

        N = self.graph.number_of_nodes()
        if self.params["init"] is not None:
            init_n_infected = ceil(N * self.params["init"])
        else:
            init_n_infected = np.random.choice(range(N))
        N = self.graph.number_of_nodes()

        n_eff = int(np.round(N * (1 - np.sqrt(1 - init_n_infected / N))))
        nodeset = np.array(list(self.graph.nodes()))
        ind1 = np.random.choice(nodeset, size=n_eff, replace=False)
        ind2 = np.random.choice(nodeset, size=n_eff, replace=False)
        ind3 = np.intersect1d(ind1, ind2)
        states = np.ones(N) * self.state_label["SS"]
        states[ind1] = self.state_label["IS"]
        states[ind2] = self.state_label["SI"]
        states[ind3] = self.state_label["II"]

        self.continue_simu = True
        return states
