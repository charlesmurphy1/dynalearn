"""

DynamicalNetwork.py

Created by Charles Murphy on 26-06-18.
Copyright © 2018 Charles Murphy. All rights reserved.
Quebec, Canada

Defines the class DynamicalNetwork which generate network on which a dynamical 
process occurs.

"""

import EoN
from random import sample
from math import ceil
import numpy as np
from DynamicalNetwork import *  

class SISNetwork_miller(DynamicalNetwork):
    """docstring for SISNetwork_miller_discrete"""
    def __init__(self, graph, inf_rate, init_active=0.01, dt=0.01,
                 filename=None):

        # Activity(Susceptible) = 0
        # Activity(Infected) = 1
        self.inf_rate = inf_rate
        self.init_active = init_active
        self.infected = set()

        super(SISNetwork_miller, self).__init__(graph, dt=dt, filename=filename)

    def _init_nodes_activity_(self):

        n = self.number_of_nodes()
        n_init_active = ceil(n * self.init_active)
        init_activity = {v:"S" for v in self.nodes()}
        ind = sample(self.nodeset, n_init_active)
        for i in ind:
            init_activity[i] = 'I'
            self.infected.add(i)

        self.t.append(0)
        return init_activity


    def update(self, step=None):

        if step is None:
            step = self.dt

        t = self.t[-1]

        sim = EoN.fast_SIS(self, self.inf_rate, 1,
                           initial_infecteds=self.infected,
                           tmin=t, tmax=t + step, 
                           return_full_data=True)

        self.t.append(sim.t()[-1])

        activity = sim.get_statuses(time=sim.t()[-1])
        self.infected = set()
        for v in self.nodes():
            if activity[v] == 'I':
                self.infected.add(v)

        self.activity = activity
        # print(sim.get_statuses(), self.inf_rate)

        return 0

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


        return {'S':np.mean(S_array), 'I':np.mean(I_array)}, \
               {'S':np.std(S_array), 'I':np.std(I_array)} 


class SIRNetwork_miller(DynamicalNetwork):
    """docstring for SIRNetwork_miller"""
    def __init__(self, graph, inf_rate, init_active=0.01, dt=0.01,
                 filename=None):

        self.inf_rate = inf_rate
        self.init_active = init_active

        self.infected = set()
        self.recovered = set()

        super(SIRNetwork_miller, self).__init__(graph, dt=dt, filename=filename)

    def _init_nodes_activity_(self):

        n = self.number_of_nodes()
        n_init_active = ceil(n * self.init_active)
        init_activity = {v:"S" for v in self.nodes()}
        ind = sample(self.nodeset, n_init_active)

        for i in ind:
            init_activity[i] = 'I'
            self.infected.add(i)

        self.t.append(0)
        return init_activity


    def update(self, step=None, save=False):

        # Default value of step
        if step is None:
            step = self.dt

        t = self.t[-1]

        # time step with Miller's code
        sim = EoN.fast_SIR(self, self.inf_rate, 1,
                           initial_infecteds=self.infected,
                           initial_recovereds=self.recovered,
                           tmin=t, tmax=t + step, 
                           return_full_data=True)

        # adding time step to time array
        self.t.append(t + step)

        # get current activity
        activity = sim.get_statuses(time=sim.t()[-1])

        # update infected and recovered
        self.infected = set()
        self.recovered = set()
        for v in self.nodes():
            if activity[v] == 'I':
                self.infected.add(v)
            elif activity[v] == 'R':
                self.recovered.add(v)

        self.activity = activity

        if self.full_data_mode:
            self.history[t] = activity

        if save:
            self.save()

        return 0

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


        return {'S':np.mean(S_array), 'I':np.mean(I_array), 'R':np.mean(R_array)}, \
               {'S':np.std(S_array), 'I':np.std(I_array), 'R':np.std(R_array)}   
        

if __name__ == '__main__':
    import networkx as nx
    import matplotlib.pyplot as plt

    g = nx.gnm_random_graph(1000,500)
    init_active = 0.1
    dt = 0.001
    step = 0.01
    T = 10
    rate = 0.5
    filename = "myfile.b"
    save = False

    def plot_activity(model_class):
        ## Testing SIRNetwork sub-class
        m_net = model_class(g, rate, init_active=init_active, dt=dt,
                             filename=filename)

        avg, err = m_net.get_avg_activity()
        t = [0]
        avg_act = [list(avg.values())]
        err_act = [list(err.values())]


        while(t[-1] < T and m_net.continu_simu):

            m_net.update(step=step, save=False)

            avg, err = m_net.get_avg_activity()

            t.append(m_net.t[-1])
            avg_act.append(list(avg.values()))
            err_act.append(list(err.values()))
        m_net.close()

        avg_act = np.array(avg_act)
        err_act = np.array(err_act)

        plt.plot(t, avg_act)
        plt.show()

    # plot_activity(SISNetwork_miller)
    plot_activity(SIRNetwork_miller)
