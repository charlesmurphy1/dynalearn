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


class DynamicsModel(ABC):
    def __init__(self, config, num_states):
        self._config = config
        self._num_states = num_states
        self._network = None
        self._adj = None
        self._edgelist = None
        self._num_nodes = None

    @abstractmethod
    def initial_state(self):
        raise NotImplementedError("self.initial_state() has not been impletemented")

    @abstractmethod
    def predict(self, x):
        raise NotImplementedError("self.transition() has not been impletemented")

    @abstractmethod
    def sample(self, x):
        """
		Computes the next activity states. (virtual) (private)

		"""
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
        self._num_nodes = self._network.number_of_nodes()

    @property
    def num_nodes(self):
        if self._num_nodes is None:
            raise ValueError("No network has been parsed to the dynamics.")
        else:
            return self._num_nodes

    @property
    def num_states(self):
        return self._num_states
