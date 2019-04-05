"""

epidemic.py

Created by Charles Murphy on 26-06-18.
Copyright © 2018 Charles Murphy. All rights reserved.
Quebec, Canada

Defines DynamicalNetwork sub-classes for epidemic spreading dynamics.

"""

import numpy as np
from math import ceil
from random import sample, random, choice

from .dynamical_network import *


class EpidemicNetwork(DynamicalNetwork):
    def __init__(self, graph, state_label, init_param):
        self.init_param = init_param

        if "S" not in state_label or "I" not in state_label:
            raise ValueError("state_label must contain states 'S' and 'I'.")
        else:
            self.state_label = state_label
            self.inv_state_label = {state_label[i]:i for i in state_label}
            self.state_nodeset = {l: set() for l in state_label}

        super(EpidemicNetwork, self).__init__(graph)

    def initialize_states(self):
        N = self.number_of_nodes()
        if self.init_param is not None:
            init_n_infected = ceil(N * self.init_param)
        else:
            init_n_infected = choice(range(N))
        nodeset = list(self.nodes())
        ind = sample(nodeset, init_n_infected)
        self.state_nodeset['S'] = set(range(N)).difference(ind)
        self.state_nodeset['I'] = set().union(ind)

        states = np.ones(N) * self.state_label['S']
        states[ind] = self.state_label['I']

        self.t = [0]

        self.continue_simu = True
        self.states = states

    def get_num_neighbors(self, node, state):

        neighbors = self.neighbors(node)
        num_neighbors = 0

        for n in neighbors:
            if self.states[n] == self.state_label[state]:
                num_neighbors += 1                

        return num_neighbors

    def get_avg_state(self):

        
        state_dict = {i:np.zeros(self.number_of_nodes()) for i in self.state_label}

        for v in self.nodes():
            l = self.inv_state_label[self.states[v]]
            state_dict[l][i] = 1

        avg_states = {l: np.mean(state_dict[l] for l in state_dict)}
        std_states = {l: np.std(state_dict[l] for l in state_dict)}
        return avg_states, std_states

    def get_state_from_value(self, state):
        inv_state_label = {self.state_label[i]:i for i in self.state_label}





class SISNetwork(EpidemicNetwork):
    """
        Class for  discrete SIS dynamical network.

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
    def __init__(self, graph, infection_prob, recovery_prob, init_param=None):

        self.infection_prob = infection_prob
        self.recovery_prob = recovery_prob
        state_label = {'S':0, 'I':1}
        
        super(SISNetwork, self).__init__(graph, state_label, init_param)


    def transition_states(self):

        states = self.states.copy()

        new_susceptible = self.state_nodeset['S'].copy()
        new_infected = self.state_nodeset['I'].copy()

        for inf in self.state_nodeset['I']:
            neighbors = self.neighbors(inf)

            for n in neighbors:
                if random() < self.infection_prob and states[n] == self.state_label['S']:
                    states[n] = self.state_label['I']
                    new_infected.add(n)
                    new_susceptible.remove(n)

            if random() < self.recovery_prob:
                states[inf] = 0
                new_susceptible.add(inf)
                new_infected.remove(inf)

        self.state_nodeset['S'] = new_susceptible
        self.state_nodeset['I'] = new_infected

        if len(self.state_nodeset['I']) == 0:
            self.continue_simu = False

        return states

    def node_transition_probability(self, node):
        num_infected = self.get_num_neighbors(node, 'I')

        if self.states[node] == self.state_label['I']:
            p_I = 1 - (1 - self.infection_prob)**num_infected
        else:
            p_I = 1 - self.recovery_prob
        
        return {'S':1 - p_I, 'I':p_I}
        

class SIRNetwork(EpidemicNetwork):
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
    def __init__(self, graph, infection_prob, recovery_prob, init_param=None):

        self.infection_prob = infection_prob
        self.recovery_prob = recovery_prob
        self.init_param = init_param
        state_label = {'S':0, 'I':1, 'R':2}
        
        super(SIRNetwork, self).__init__(graph, state_label, init_param)


    def transition_states(self):

        states = self.states.copy()

        new_susceptible = self.state_nodeset['S'].copy()
        new_infected = self.state_nodeset['I'].copy()
        new_recovered = self.state_nodeset['R'].copy()

        for inf in self.state_nodeset['I']:
            neighbors = self.neighbors(inf)

            for n in neighbors:
                if random() < self.infection_prob and states[n] == self.state_label['S']:
                    states[n] = self.state_label['I']
                    new_infected.add(n)
                    new_susceptible.remove(n)

            if random() < self.recovery_prob:
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
        num_infected = self.get_num_neighbors(states, 'I')

        if self.states[node] == self.state_label['S']:
            p_I = 1 - (1 - self.infection_prob)**num_infected
            p_R = 0
        elif self.states[node] == self.state_label['I']:
            p_I = 1 - self.recovery_prob
            p_R = self.recovery_prob
        
        return {'S':1 - p_I - p_R, 'I':p_I, 'R':p_R}
               