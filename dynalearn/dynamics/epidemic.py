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


class EpidemicDynamics(Dynamics):
    def __init__(self, state_label, init_state):

        super(EpidemicDynamics, self).__init__()

        if "S" not in state_label or "I" not in state_label:
            raise ValueError("state_label must contain states 'S' and 'I'.")
        else:
            self.state_label = state_label
            self.inv_state_label = {state_label[i]:i for i in state_label}
        self.init_state = init_state

        for name, value in self.state_label.items():
            self.params["state_" + name] = value



    def initialize_states(self):
        N = self.graph.number_of_nodes()
        if self.init_state is not None:
            init_n_infected = ceil(N * self.init_state)
        else:
            init_n_infected = np.random.choice(range(N))
        nodeset = np.array(self.graph.nodes())
        ind = np.random.choice(nodeset, size=init_n_infected, replace=False)
        states = np.ones(N) * self.state_label['S']
        states[ind] = self.state_label['I']

        self.t = [0]

        self.continue_simu = True
        self.states = states


    def transition(self):

        p = self.predict(self.states)
        new_states = np.random.binomial(1, p)
        if np.sum(new_states) == 0:
            self.continue_simu = False

        return new_states

    def infected_degrees(self, states):
        N = self.graph.number_of_nodes()
        if len(states.shape) < 2:
            states = states.reshape(1, N)
        adj = nx.to_numpy_array(self.graph)
        return np.matmul(states, adj)

    def get_avg_state(self):
        N = self.graph.number_of_nodes()
        state_dict = {l:np.zeros(N) for l in self.state_label}

        for v in self.graph.nodes():
            label = self.inv_state_label[self.states[v]]
            state_dict[label][v] = 1

        avg_states = {l: np.mean(state_dict[l]) for l in state_dict}
        std_states = {l: np.std(state_dict[l]) for l in state_dict}
        return avg_states, std_states

    def get_state_from_value(self, state):
        inv_state_label = {self.state_label[i]:i for i in self.state_label}


    def estimate_cltp(self, in_states, out_states):
        N = self.graph.number_of_nodes()
        if len(in_states.shape) < 2:
            states = states.reshape(1, N)

        inf_deg = self.infected_degrees(in_states)

        avg_prob = {}
        var_prob = {}

        for l in range(int(np.max(inf_deg))):
            for in_s in self.state_label:
                for out_s in self.state_label:
                    in_condition = np.where(np.logical_and(states == in_s,
                                                           inf_deg == l),
                                            np.ones(N),
                                            np.zeros(N))

                    avail_state = out_states[in_condition == 1]
                    n_sample = len(avail_state)
                    out_condition = np.zeros(n_sample)
                    out_condition[avail_state == out_s] = 1

                    avg_prob[(in_s, out_s)] = np.mean(out_condition)
                    var_prob[(in_s, out_s)] = np.std(out_condition) / np.sqrt(n_sample)

        return avg_prob, var_prob


class SISDynamics(EpidemicDynamics):
    """
        Class for  discrete SIS dynamics.

        **Parameters**
        graph : nx.Graph
            A graph on which the dynamical process occurs.

        infection_prob : Float
            Infection probability.

        recovery_prob : Float
            Recovery probability.

        filename : String : (default = ``None``)
            Name of file for saving states. If ``None``, it does not save the states.

    """
    def __init__(self, infection_prob, recovery_prob, init_state=None):
        super(SISDynamics, self).__init__({'S':0, 'I':1},
                                         init_state)

        self.params["infection_prob"] = infection_prob
        self.params["recovery_prob"] = recovery_prob


    def predict(self, states):
        N = self.graph.number_of_nodes()
        if len(states.shape) < 2:
            states = states.reshape(1, N)

        beta = self.params["infection_prob"]
        alpha = self.params["recovery_prob"]

        inf_deg = self.infected_degrees(states)

        state_prob = np.zeros(states.shape)
        state_prob[states == 0] = 1 - (1 - beta)**inf_deg[states==0]
        state_prob[states == 1] = 1 - alpha

        return state_prob


    def cond_local_trans_prob(self):
        N = self.graph.number_of_nodes()
        p_s = 1 - (1 - self.infection_prob)**np.arange(N)
        p_i = np.ones(N) * (1 - self.recovery_prob)

        return {'S': p_s, 'I': p_i}


class SIRDynamics(EpidemicDynamics):
    """
        Class for  discrete SIR dynamical network.

        **Parameters**
        graph : nx.Graph
            A graph on which the dynamical process occurs.

        infection_prob : Float
            Infection probability.

        recovery_prob : Float
            Recovery probability.

        filename : String : (default = ``None``)
            Name of file for saving states. If ``None``, it does not save the states.

    """
    def __init__(self, graph, infection_prob, recovery_prob, init_state=None):
        super(SIRDynamics, self).__init__({'S':0, 'I':1, 'R':-1},
                                         init_state)
        
        self.params["infection_prob"] = infection_prob
        self.params["recovery_prob"] = recovery_prob


    def transition_states(self):

        states = self.states.copy()

        new_susceptible = self.state_nodeset['S'].copy()
        new_infected = self.state_nodeset['I'].copy()
        new_recovered = self.state_nodeset['R'].copy()

        for inf in self.state_nodeset['I']:
            neighbors = self.graph.neighbors(inf)

            for n in neighbors:
                if random() < self.params["infection_prob"] and states[n] == self.state_label['S']:
                    states[n] = self.state_label['I']
                    new_infected.add(n)
                    new_susceptible.remove(n)

            if random() < self.params["recovery_prob"]:
                states[inf] = self.state_label['R']
                new_recovered.add(inf)
                new_infected.remove(inf)

        self.state_nodeset['S'] = new_susceptible
        self.state_nodeset['I'] = new_infected
        self.state_nodeset['R'] = new_recovered

        if len(self.state_nodeset['I']) == 0:
            self.continue_simu = False

        return states

    def node_transition_probability(self, node):
        num_infected = self.get_num_neighbors(self.graph, states, 'I')

        if self.states[node] == self.state_label['S']:
            p_I = 1 - (1 - self.params["infection_prob"])**num_infected
            p_R = 0
        elif self.states[node] == self.state_label['I']:
            p_I = 1 - self.params["recovery_prob"]
            p_R = self.params["recovery_prob"]
        
        return {'S':1 - p_I - p_R, 'I':p_I, 'R':p_R}
               