"""

epidemic.py

Created by Charles Murphy on 26-06-18.
Copyright © 2018 Charles Murphy. All rights reserved.
Quebec, Canada

Defines DynamicalNetwork sub-classes for epidemic spreading dynamics.

"""

import numpy as np
from math import ceil
from dynalearn.dynamics import *


class Epidemics(Dynamics):
    def __init__(self, state_label, init_state):
        super(Epidemics, self).__init__(len(state_label))
        self.state_label = state_label
        self.inv_state_label = {state_label[i]: i for i in state_label}
        self.init_state = init_state

        for name, value in self.state_label.items():
            self.params["state_" + name] = value

    def state_degree(self, states, adj=None):
        if adj is None:
            adj = nx.to_numpy_array(self.graph)
        N = adj.shape[0]
        if len(states.shape) < 2:
            states = states.reshape(1, N)

        state_l = {
            s: np.matmul(states == self.state_label[s], adj).squeeze()
            for s in self.state_label
        }

        return state_l

    def get_avg_state(self):
        N = self.graph.number_of_nodes()
        state_dict = {l: np.zeros(N) for l in self.state_label}

        for v in self.graph.nodes():
            label = self.inv_state_label[self.states[v]]
            state_dict[label][v] = 1

        avg_states = {l: np.mean(state_dict[l]) for l in state_dict}
        std_states = {l: np.std(state_dict[l]) for l in state_dict}
        return avg_states, std_states


class SingleEpidemics(Epidemics):
    def __init__(self, state_label, init_state):

        if "S" not in state_label or "I" not in state_label:
            raise ValueError("state_label must contain states 'S' and 'I'.")
        super(SingleEpidemics, self).__init__(state_label, init_state)

    def initialize_states(self):
        N = self.graph.number_of_nodes()
        if self.init_state is not None:
            init_n_infected = ceil(N * self.init_state)
        else:
            init_n_infected = np.random.choice(range(N))
        nodeset = np.array(list(self.graph.nodes()))
        ind = np.random.choice(nodeset, size=init_n_infected, replace=False)
        states = np.ones(N) * self.state_label["S"]
        states[ind] = self.state_label["I"]

        self.t = [0]

        self.continue_simu = True
        self.states = states
        return states


class DoubleEpidemics(Epidemics):
    def __init__(self, state_label, init_state):
        if (
            "SS" not in state_label
            or "SI" not in state_label
            or "IS" not in state_label
            or "II" not in state_label
        ):
            raise ValueError("state_label must contain states 'S' and 'I'.")
        super(DoubleEpidemics, self).__init__(state_label, init_state)

    def initialize_states(self):
        N = self.graph.number_of_nodes()
        if self.init_state is not None:
            init_n_infected = ceil(N * self.init_state)
        else:
            init_n_infected = np.random.choice(range(N))

        n_eff = int(np.round(N * (1 - np.sqrt(1 - init_n_infected / N))))
        nodeset = np.array(list(self.graph.nodes()))
        ind1 = np.random.choice(nodeset, size=n_eff, replace=False)
        ind2 = np.random.choice(nodeset, size=n_eff, replace=False)
        ind3 = np.intersect1d(ind1, ind2)
        states = np.ones(N) * self.state_label["SS"]
        states[ind1] = self.state_label["IS"]
        states[ind2] = self.state_label["SI"]
        states[ind3] = self.state_label["II"]

        self.t = [0]

        self.continue_simu = True
        self.states = states
