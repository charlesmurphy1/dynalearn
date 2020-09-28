"""

dynamic.py

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
from dynalearn.utilities import (
    to_edge_index,
    get_edge_attr,
    get_node_strength,
    collapse_networks,
)


class Dynamics(ABC):
    def __init__(self, config, num_states):
        self._config = config
        self._num_states = num_states
        self._network = None
        self._edge_index = None
        self._num_nodes = None

    @abstractmethod
    def initial_state(self):
        raise NotImplementedError("self.initial_state() has not been impletemented")

    @abstractmethod
    def predict(self, x):
        raise NotImplementedError("self.predict() has not been impletemented")

    @abstractmethod
    def loglikelihood(self, x):
        raise NotImplementedError("self.loglikelihood() has not been impletemented")

    @abstractmethod
    def sample(self, x):
        raise NotImplementedError("sample has not been impletemented")

    @abstractmethod
    def is_dead(self, x):
        raise NotImplementedError("is_dead has not been impletemented")

    @property
    def network(self):
        if self._network is None:
            raise ValueError("No network has been parsed to the dynamics.")
        else:
            return self._network

    @network.setter
    def network(self, network):
        self._network = network
        if not network.is_directed():
            network = nx.to_directed(network)
        self._edge_index = to_edge_index(network)
        self._node_degree = np.array(list(dict(self.network.degree()).values()))
        self._num_nodes = self._network.number_of_nodes()

    @property
    def edge_index(self):
        if self._edge_index is None:
            raise ValueError("No network has been parsed to the dynamics.")
        else:
            return self._edge_index

    @property
    def node_degree(self):
        if self._node_degree is None:
            raise ValueError("No network has been parsed to the dynamics.")
        else:
            return self._node_degree

    @property
    def num_nodes(self):
        if self._num_nodes is None:
            raise ValueError("No network has been parsed to the dynamics.")
        else:
            return self._num_nodes

    @property
    def num_states(self):
        return self._num_states


class WeightedDynamics(Dynamics):
    def __init__(self, config, num_states):
        Dynamics.__init__(self, config, num_states)
        self._edge_attr = None
        self._node_strength = None

    @property
    def network(self):
        if self._network is None:
            raise ValueError("No network has been parsed to the dynamics.")
        else:
            return self._network

    @network.setter
    def network(self, network):
        self._network = network
        if not nx.is_directed(network):
            network = nx.to_directed(network)
        self._edge_index = to_edge_index(network)
        self._edge_weight = get_edge_attr(network)["weight"].reshape(-1, 1)
        self._node_degree = np.array(list(dict(self.network.degree()).values()))
        self._node_strength = get_node_strength(network).reshape(-1, 1)
        self._num_nodes = self._network.number_of_nodes()

    @property
    def edge_weight(self):
        if self._edge_weight is None:
            raise ValueError("No network has been parsed to the dynamics.")
        else:
            return self._edge_weight

    @property
    def node_strength(self):
        if self._node_strength is None:
            raise ValueError("No network has been parsed to the dynamics.")
        else:
            return self._node_strength


class MultiplexDynamics(Dynamics):
    @property
    def network(self):
        if self._network is None:
            raise ValueError("No network has been parsed to the dynamics.")
        else:
            return self._network

    @network.setter
    def network(self, network):
        self._network = network
        self._edge_index = {}
        self._node_degree = {}
        self._num_nodes = None
        self._num_networks = len(self._network)
        for k, net in network.items():
            if not net.is_directed():
                net = nx.to_directed(net)
            if self._num_nodes is None:
                self._num_nodes = net.number_of_nodes()
            else:
                assert self._num_nodes == net.number_of_nodes()
            self._node_degree[k] = np.array(
                list(dict(self._network[k].degree()).values())
            )
            self._edge_index[k] = to_edge_index(net)
        if "all" not in network:
            self._network["all"] = collapse_networks(network)
        else:
            self._network["all"] = network["all"]
        self._edge_index["all"] = to_edge_index(self._network["all"])


class WeightedMultiplexDynamics(Dynamics):
    def __init__(self, config, num_states):
        Dynamics.__init__(self, config, num_states)
        self._edge_attr = None
        self._node_strength = None

    @property
    def network(self):
        if self._network is None:
            raise ValueError("No network has been parsed to the dynamics.")
        else:
            return self._network

    @network.setter
    def network(self, network):
        self._network = network
        self._edge_index = {}
        self._node_degree = {}
        self._edge_weight = {}
        self._node_strength = {}
        self._num_nodes = None
        self._num_networks = len(self._network)
        for k, net in network.items():
            if not net.is_directed():
                net = nx.to_directed(net)
            if self._num_nodes is None:
                self._num_nodes = net.number_of_nodes()
            else:
                assert self._num_nodes == net.number_of_nodes()
            self._node_degree[k] = np.array(
                list(dict(self._network[k].degree()).values())
            )
            self._edge_index[k] = to_edge_index(net)
            self._edge_weight[k] = get_edge_attr(net)["weight"]
            self._node_strength[k] = get_node_strength(net)
        if "all" not in network:
            self._network["all"] = collapse_networks(network)
        else:
            self._network["all"] = network["all"]
        self._edge_index["all"] = to_edge_index(self._network["all"])
        self._node_degree["all"] = np.array(
            list(dict(self._network["all"].degree()).values())
        )
        self._edge_weight["all"] = get_edge_attr(self._network["all"])["weight"]
        self._node_strength["all"] = get_node_strength(self._network["all"])

    @property
    def edge_weight(self):
        if self._edge_weight is None:
            raise ValueError("No network has been parsed to the dynamics.")
        else:
            return self._edge_weight

    @property
    def node_strength(self):
        if self._node_strength is None:
            raise ValueError("No network has been parsed to the dynamics.")
        else:
            return self._node_strength
