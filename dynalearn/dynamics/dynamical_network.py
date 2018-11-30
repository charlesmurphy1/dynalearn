"""

dynamical_network.py

Created by Charles Murphy on 26-06-18.
Copyright © 2018 Charles Murphy. All rights reserved.
Quebec, Canada

Defines the class Dynamical_Network which generate network on which a dynamical 
process occurs.

"""

import networkx as nx
import pickle

__all__ = ['Dynamical_Network']

class Dynamical_Network(nx.Graph):
	"""
		Base class for dynamical network.

		**Parameters**
		graph : nx.Graph
			A graph on which the dynamical process occurs.

		dt : Float : (default = ``0.01``)
			Time step.

		filename : String : (default = ``None``)
			Name of file for saving activity states. If ``None``, it does not save the states.

		full_data_mode : Bool : (default = ``False``)
			

	"""
	def __init__(self, graph, dt=0.01, filename=None, full_data_mode=False):
		"""
		Initializes a Dynamical_Network object.

		"""
		super(Dynamical_Network, self).__init__(graph)
		self.nodeset = [v for v in self.nodes()]
		self.edgeset = [e for e in self.edges()]

		if dt <= 0:
			raise ValueError("dt must be greater than 0.")
		else:
			self.dt = dt

		self.continu_simu = True

		self.t = []
		self.activity = self._init_nodes_activity()

		self.full_data_mode = full_data_mode

		if filename is None:
			self.saving_file = None
		else:
			self.saving_file = open(filename, "wb")


	def _init_nodes_activity_(self):
		"""
		Initializes the nodes activity states. (virtual) (private)

		"""
		raise NotImplementedError("self._init_nodes_activity_() has not been impletemented")


	def _state_transition_(self):
		"""
		Computes the next activity states. (virtual) (private)

		"""
		raise NotImplementedError("self._state_transition_() has not been impletemented")	


	def update(self, step=None, record=False):
		"""
		Update the next activity states.

		**Parameters**
		step : Integer : (default = ``None``)
			Number of steps to perform for the update.

		save : Bool : (default = ``False``)
			If ``True``, it saves the update.

		"""
		if step is None:
			step = self.dt
		t_init = self.t[-1]
		t = t_init

		while(t < t_init + step) and self.continu_simu:

			forward_activity = self._state_transition()
			self.activity = forward_activity.copy()


			t += self.dt
		
		self.t.append(t)

		if record:
			# self.history[t] = forward_activity.copy()
			self.save()


		return 0

	def get_activity(self, v=None):
		"""
		Get the activity state of node.

		**Parameters**
		v : Node key : (default = ``None``)

		**Returns**
		activity : Activity (array)
			If v is ``None``, activity is an array of all activities.

		"""
		if v is None:
			activity = self.activity.copy()
		else:
			activity = self.activity[v]

		return activity


	def get_avg_activity(self):
		"""
		Get the average activity state. (virtual)

		**Returns**
		avg_activity : Activity

		"""
		NotImplementedError("self.get_avg_activity has not been implemented.")
		return 0


	def save(self):
		"""
		Save the activity states.

		"""

		if self.saving_file is None:
			raise NameError('In Dynamical_Network object -> \
							missing _saving_file member to save.')

		if self.full_data_mode:
			pickle.dump([self.t[-1], self.activity], self.saving_file)
		else:
			avg_activity = self.get_avg_activity()
			pickle.dump([self.t[-1], avg_activity], self.saving_file)


	def load(self, f):
		"""
		Save the activity states.

		"""


		if self.saving_file is None:
			raise NameError('In Dynamical_Network object -> \
							missing _saving_file member to save.')

		pickle.dump([self.t[-1], self.activity], self.saving_file)


	def close(self):
		"""
		Close file for the activity states.

		"""
		self.saving_file.close()

