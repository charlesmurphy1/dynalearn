"""

epidemic.py

Created by Charles Murphy on 26-06-18.
Copyright © 2018 Charles Murphy. All rights reserved.
Quebec, Canada

Defines Dynamical_Network sub-classes for epidemic spreading dynamics.

"""

import numpy as np
from math import ceil
from random import sample, random

from .dynamical_network import *

__all__ = ['SISNetwork', 'SIRNetwork']

class SISNetwork(Dynamical_Network):
    """
        Class for SIS dynamical network.

        **Parameters**
        graph : nx.Graph
            A graph on which the dynamical process occurs.

        rate : Float
            Normalized infection rate.

        init_active : Float
            Initial fraction of infected nodes.

        dt : Float : (default = ``0.01``)
            Time step.

        filename : String : (default = ``None``)
            Name of file for saving activity states. If ``None``, it does not save the states.

    """
    def __init__(self, graph, rate, init_active=0.01, dt=0.01,
                 filename=None, full_data_mode=False, overwrite=True):

        self.rate = rate
        self.init_active = init_active
        self.infected = set()
        super(SISNetwork, self).__init__(graph, dt=dt, filename=filename,
                                         full_data_mode=full_data_mode,
                                         overwrite=overwrite)


    def init_activity(self):

        n = self.number_of_nodes()
        n_init_active = ceil(n * self.init_active)
        activity = {v:'S' for v in self.nodes()}
        self.t.append(0)

        if n_init_active > 0:
            n_init_active = 1
            ind = sample(self.nodeset, n_init_active)
            self.infected = self.infected.union(ind)
            for i in ind:
                activity[i] = 'I'

        return activity


    def _state_transition(self):

        activity = self.activity.copy()

        new_infected = self.infected.copy()
        for inf in self.infected:

            neighbors = self.neighbors(inf)

            for n in neighbors:
                r = random()
                if r < self.rate * self.dt:
                    activity[n] = 'I'
                    new_infected.add(n)
            r = random()
            if r < self.dt:
                activity[inf] = 'S'
                new_infected.remove(inf)

        self.infected = new_infected

        if len(new_infected) == 0:
            self.continue_simu = False

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



        

class SIRNetwork(Dynamical_Network):
    """docstring for SIRNetwork"""
    def __init__(self, g, rate, init_active=0.01, dt=0.01,
                 filename=None, full_data_mode=False, overwrite=True):

        self.rate = rate
        self.init_active = init_active

        self.infected = set()

        super(SIRNetwork, self).__init__(g, dt=dt, filename=filename,
                                         full_data_mode=full_data_mode,
                                         overwrite=overwrite)


    def init_activity(self):

        n = self.number_of_nodes()
        n_init_active = ceil(n * self.init_active)
        activity = {v:'S' for v in self.nodes()}
        ind = sample(self.nodeset, n_init_active)
        self.infected = self.infected.union(ind)
        for i in ind:
            activity[i] = 'I'

        self.t.append(0)

        return activity

    def _state_transition(self):

        activity = self.activity.copy()
        new_infected = self.infected.copy()

        for inf in self.infected:
            neighbors = self.neighbors(inf)

            r = random()
            if r < self.dt:
                activity[inf] = 'R'
                new_infected.remove(inf)

            for n in neighbors:
                r = random()
                if activity[n] == 'S' and r < self.rate * self.dt:
                    activity[n] = 'I'
                    new_infected.add(n)


        self.infected = new_infected

        if len(new_infected) == 0:
            self.continue_simu = False

        return activity


    def get_avg_activity(self):
        activity_array = np.array(self.activity.values())

        S_array = np.zeros(self.number_of_nodes())
        I_array = np.zeros(self.number_of_nodes())
        R_array = np.zeros(self.number_of_nodes())

        for i, v in enumerate(self.nodes()):
            if self.activity[v] == 'S':
                S_array[i] = 1
            elif self.activity[v] == 'I':
                I_array[i] = 1
            else:
                R_array[i] = 1


        return {'S':np.mean(S_array),
                'I':np.mean(I_array),
                'R':np.mean(R_array)}, \
               {'S':np.std(S_array),
                'I':np.std(I_array),
                'R':np.std(R_array)}      


if __name__ == '__main__':
    import networkx as nx
    import matplotlib.pyplot as plt

    g = nx.gnm_random_graph(10,50)
    init_active = 0.3
    dt = 0.001
    step = 1
    T = 5
    rate = 0.5
    save = True

    def plot_activity(model_class, file_states, file_net):
        ## Testing SIRNetwork sub-class

        if save:
            net_file = open(file_net, "wb")
            nx.write_edgelist(g, net_file)
            net_file.close()
        else:
            file_states = None

        m_net = model_class(g, rate, init_active=init_active, dt=dt,
                             filename=file_states)

        avg, err = m_net.get_avg_activity()
        t = [0]
        avg_act = [list(avg.values())]
        err_act = [list(err.values())]


        while(t[-1] < T and m_net.continue_simu):

            m_net.update(step=step, save=save)

            avg, err = m_net.get_avg_activity()

            t.append(m_net.t[-1])
            avg_act.append(list(avg.values()))
            err_act.append(list(err.values()))

        if save:
            m_net.close()

        avg_act = np.array(avg_act)
        err_act = np.array(err_act)

        plt.plot(t, avg_act)
        plt.show()

    plot_activity(SISNetwork, "testdata/SIS_states.b", "testdata/SIS_net.txt")
    plot_activity(SIRNetwork, "testdata/SIR_states.b", "testdata/SIR_net.txt")

