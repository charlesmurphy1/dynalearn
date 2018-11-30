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


class DSISNetwork(Dynamical_Network):
    """
        Class for D-SIS dynamical network.

        **Parameters**
        graph : nx.Graph
            A graph on which the dynamical process occurs.

        rate : float
            Normalized infection rate.

        D : int
            Number of diseases.

        in_coupling : float
            Coupling constant for recovery.

        out_coupling : float
            Coupling constant for infection.

        K : int
            Number of intermediate states.

        init_active : Float
            Initial fraction of infected nodes.

        dt : Float : (default = ``0.01``)
            Time step.

        filename : String : (default = ``None``)
            Name of file for saving activity states. If ``None``, it does not save the states.

    """
    def __init__(self, graph, inf_rate, rec_rate, D, in_coupling, out_coupling,
                 init_active=0.01, init_num=1, dt=0.01, filename=None,
                 full_data_mode=False):

        self.inf_rate = inf_rate
        self.rec_rate = rec_rate
        self.D = D
        self.in_coupling = in_coupling
        self.out_coupling = out_coupling
        self.init_active = init_active
        self.init_num = init_num
        self.infected = set()

        self.disease_list = list(range(self.D))

        super(DSISNetwork, self).__init__(graph, dt=dt, filename=filename,
                                         full_data_mode=full_data_mode)


    def _init_nodes_activity(self):

        n = self.number_of_nodes()
        n_init_active = ceil(n * self.init_active)
        init_activity = {v:np.zeros(self.D) for v in self.nodes()}
        ind = sample(self.nodeset, n_init_active)

        self.infected = self.infected.union(ind)

        p = self.init_num / self.D
        if p >= 1: p = 1 - 1e-15
        for i in ind:
            init_activity[i] = np.random.binomial(1, p, self.D)

        self.t.append(0)

        return init_activity


    def _state_transition(self):

        activity = self.activity.copy()

        new_infected = self.infected.copy()

        for inf in self.infected:

            neighbors = self.neighbors(inf)
            for n in neighbors:
                num = np.dot(self.activity[n], 1 - self.activity[inf])
                p = self.inf_rate * self.dt
                for d in range(self.D):
                    if random() < p and self.activity[inf][d] == 1:
                        activity[n][d] = 1
                        new_infected.add(n)
            
            num = np.dot(self.activity[inf], self.activity[inf])
            p = self.rec_rate * self.in_coupling**num * self.dt
            for d in range(self.D):
                if random() < p: activity[inf][d] = 0
            if np.sum(activity[inf]) == 0: new_infected.remove(inf)

        self.infected = new_infected

        if len(new_infected) == 0:
            self.continu_simu = False

        return activity

    def get_avg_activity(self):
        activity_array = np.array(self.activity.values())

        
        S_array = np.zeros(self.number_of_nodes())
        I_array = np.zeros([self.number_of_nodes(), self.D])

        nS = 0
        nI = 0
        for i, v in enumerate(self.nodes()):
            num = int(np.sum(self.activity[v]))

            if num < 1:
                S_array[i] = 1
                nS += 1
            else:
                nI += 1
                I_array[i, num - 1] = 1
        all_I_array = np.sum(I_array, axis=1)
        all_I_array[all_I_array>1] = 1

        avg = {'S': np.mean(S_array), 'I_{tot}': np.mean(all_I_array)}
        std = {'S': np.std(S_array), 'I_{tot}': np.std(all_I_array)}

        for i in range(self.D):
            avg[f'n = {i + 1}'] = np.mean(I_array[:, i])
            std[f'n = {i + 1}'] = np.std(I_array[:, i])

        return avg, std

