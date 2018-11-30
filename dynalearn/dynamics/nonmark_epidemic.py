"""

epidemic.py

Created by Charles Murphy on 26-06-18.
Copyright © 2018 Charles Murphy. All rights reserved.
Quebec, Canada

Defines Dynamical_Network sub-classes for complex epidemic spreading dynamics.

"""

import numpy as np
from math import ceil
from random import sample, random

from .dynamical_network import *

__all__ = ['SIkSNetwork']


class SIkSNetwork(Dynamical_Network):
    """
        Class for SIkS dynamical network.

        **Parameters**
        graph : nx.Graph
            A graph on which the dynamical process occurs.

        rate : float
            Normalized infection rate.

        K : int
            Number of intermediate states.

        init_active : Float
            Initial fraction of infected nodes.

        dt : Float : (default = ``0.01``)
            Time step.

        filename : String : (default = ``None``)
            Name of file for saving activity states. If ``None``, it does not save the states.

    """
    def __init__(self, graph, rate, K, init_active=0.01, dt=0.01,
                 filename=None, full_data_mode=False):

        self.rate = rate
        self.K = K
        self.init_active = init_active
        self.infected = set()

        self.infected_time = {}
        super(SIkSNetwork, self).__init__(graph, dt=dt, filename=filename,
                                         full_data_mode=full_data_mode)


    def _init_nodes_activity(self):

        n = self.number_of_nodes()
        n_init_active = ceil(n * self.init_active)
        init_activity = {v:'S' for v in self.nodes()}
        ind = sample(self.nodeset, n_init_active)

        self.infected = self.infected.union(ind)
        self.infected_time = {v:0 for v in self.nodes()}

        for i in ind:
            init_activity[i] = 'I'
            self.infected_time[i] += 1

        self.t.append(0)

        return init_activity


    def get_infected_neighbors(self, node):
        infected_neighbors = set()

        neighbors = self.neighbors(node)
        for n in neighbors:
            if self.activity[n] == "I":
                infected_neighbors.add(n)

        return infected_neighbors


    def _state_transition(self):

        activity = self.activity.copy()

        new_infected = self.infected.copy()

        for inf in self.infected:

            neighbors = self.neighbors(inf)
            for n in neighbors:
                
                if random() < self.rate * self.dt:
                    activity[n] = 'I'
                    self.infected_time[n] += 1
                    new_infected.add(n)

            if random() < self.dt:
                self.infected_time[n] += 1
                if self.infected_time[n] > self.K:
                    activity[inf] = 'S'
                    new_infected.remove(inf)

        self.infected = new_infected

        if len(new_infected) == 0:
            self.continu_simu = False

        return activity

    def get_avg_activity(self):
        activity_array = np.array(self.activity.values())

        
        S_array = np.zeros(self.number_of_nodes())
        I_array = np.zeros(self.number_of_nodes())

        for i, v in enumerate(self.nodes()):
            if self.activity[v] == 'S':
                S_array[i] = 1
            else:
                I_array[i] = 1

        return {'S':np.mean(S_array), 'I':np.mean(I_array)}, \
               {'S':np.std(S_array), 'I':np.std(I_array)}

