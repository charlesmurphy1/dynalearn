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
import os


class Dynamics:
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
        self._degree = None
        self.params = dict()

    def initialize_states(self):
        """
		Initializes the nodes activity states. (virtual) (private)

		"""
        raise NotImplementedError("self.initialize_states() has not been impletemented")

    def predict(self, states=None, ajd=None):
        """
		Computes the next activity states probability distribution. (virtual) (private)

		"""
        raise NotImplementedError("self.transition() has not been impletemented")

    def update(self, states=None, ajd=None):
        """
		Computes the next activity states. (virtual) (private)

		"""
        raise NotImplementedError("self.transition() has not been impletemented")

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

    @property
    def degree(self):
        if self._degree is None:
            raise ValueError("No graph has been parsed to the dynamics.")
        else:
            return self._degree

    @graph.setter
    def graph(self, graph):
        self._graph = graph
        self._degree = np.sum(nx.to_numpy_array(graph), axis=1)
        self.initialize_states()

    @property
    def num_states(self):
        return self._num_states
