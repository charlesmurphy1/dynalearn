"""

DynamicalNetwork.py

Created by Charles Murphy on 26-06-18.
Copyright © 2018 Charles Murphy. All rights reserved.
Quebec, Canada

Defines the class DynamicalNetwork which generate network on which a dynamical 
process occurs.

"""

import networkx as nx
import pickle

def transition_zero(v, t, dt):
	return 0


class DynamicalNetwork(nx.Graph):
	"""
		The class DynamicalNetwork generates a dynamical network where each 
		node has a time-dependent activity.

		Parameters
		----------
		g : nx.Graph
			A graph on which the dynamical process occurs (if None, the class
			defines an empty graph)
		dt : float
			Time step of the process
		dim : int
			Dimension of the activity vector
		transition


	"""
	def __init__(self, graph, dt=0.01, filename=None, full_data_mode=False):

		super(DynamicalNetwork, self).__init__(graph)

		# if type(graph) is not nx.classes.graph.Graph:
		# 	raise NameError("Init of DynamicalNetwork -> \
		# 					graph must be nx.Graph type.")
		# else:
		# 	self.add_nodes_from(graph.nodes())
		# 	self.add_edges_from(graph.edges())

		self.nodeset = [v for v in self.nodes()]
		self.edgeset = [e for e in self.edges()]

		if dt <= 0:
			raise NameError("Init of DynamicalNetwork -> dt must be greater than 0.")
		else:
			self.dt = dt

		self.continu_simu = True

		self.t = []
		self.activity = self._init_nodes_activity()

		self.full_data_mode = full_data_mode
		self.history = {}

		if filename is not None:
			self.saving_file = open(filename, "wb")


	def _init_nodes_activity_(self):

		raise NotImplementedError("self._init_nodes_activity_ has not been impletemented")


	def _state_transition_(self):

		raise NotImplementedError("self._state_transition_ has not been impletemented")	


	def update(self, step=None, save=False):

		if step is None:
			step = self.dt
		t_init = self.t[-1]
		t = t_init

		while(t < t_init + step) and self.continu_simu:

			forward_activity = self._state_transition()
			self.activity = forward_activity.copy()


			t += self.dt
		
		self.t.append(t)

		if self.full_data_mode:
			self.history[t] = forward_activity.copy()

		if save:
			self.save()

		return 0

	def get_activity(self, v=None):

		if v is None:
			ans = self.activity.copy()
		else:
			ans = self.activity[v]

		return ans


	def get_avg_activity(self):
		NotImplementedError("self.get_avg_activity has not been implemented.")
		return 0


	def save(self):

		if self.saving_file is None:
			raise NameError('In DynamicalNetwork object -> \
							missing _saving_file member to save.')

		pickle.dump([self.t[-1], self.activity], self.saving_file)


	def close(self):
		self.saving_file.close()

