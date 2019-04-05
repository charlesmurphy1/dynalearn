"""

dynamical_network.py

Created by Charles Murphy on 26-06-18.
Copyright © 2018 Charles Murphy. All rights reserved.
Quebec, Canada

Defines the class DynamicalNetwork which generate network on which a dynamical 
process occurs.

"""

import networkx as nx
import pickle
import os


class DynamicalNetwork(nx.Graph):
	"""
		Base class for dynamical network.

		**Parameters**
		graph : nx.Graph
			A graph on which the dynamical process occurs.

		filename : String : (default = ``None``)
			Name of file for saving activity states. If ``None``, it does not save the states.

		full_data_mode : Bool : (default = ``False``)
			

	"""
	def __init__(self, graph):
		"""
		Initializes a DynamicalNetwork object.

		"""
		super(DynamicalNetwork, self).__init__(graph)
		self.initialize_states()


	def initial_states(self):
		"""
		Initializes the nodes activity states. (virtual) (private)

		"""
		raise NotImplementedError("self.initial_states() has not been impletemented")


	def transition_states(self):
		"""
		Computes the next activity states. (virtual) (private)

		"""
		raise NotImplementedError("self.transition_states() has not been impletemented")	


	def update(self, step=1):
		"""
		Update the next activity states.

		**Parameters**
		step : Integer : (default = ``None``)
			Number of steps to perform for the update.

		save : Bool : (default = ``False``)
			If ``True``, it saves the update.

		"""
		for t in range(self.t[-1] + 1, self.t[-1] + 1 + step):

			forward_states = self.transition_states()
			self.states = forward_states.copy()
		self.t.append(t)

		return self.states


	def get_avg_states(self):
		"""
		Get the average states. (virtual)

		**Returns**
		avg_state : Activity

		"""
		NotImplementedError("self.get_avg_states has not been implemented.")
		return 0
