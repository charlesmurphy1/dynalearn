from .epidemics import *
import networkx as nx
import numpy as np


class SIS(SingleEpidemics):
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

    def __init__(self, params):
        super(SIS, self).__init__(params, {"S": 0, "I": 1})

    def update(self, states=None, adj=None):
        if states is None:
            states = self.states * 1
        if adj is None:
            adj = nx.to_numpy_array(self.graph)
        beta = self.params["infection"]
        alpha = self.params["recovery"]
        inf_deg = self.state_degree(states, adj)["I"]
        inf_prob = 1 - (1 - beta) ** inf_deg
        rec_prob = alpha
        new_states = states * 1

        new_states[(states == 0) * (np.random.rand(*states.shape) < inf_prob)] = 1
        new_states[(states == 1) * (np.random.rand(*states.shape) < rec_prob)] = 0

        if np.sum(new_states == self.state_label["I"]) == 0:
            self.continue_simu = False

        self.states = new_states * 1
        return new_states

    def predict(self, states=None, adj=None):
        if states is None:
            states = self.states
        if adj is None:
            adj = nx.to_numpy_array(self.graph)
        beta = self.params["infection"]
        alpha = self.params["recovery"]
        inf_deg = self.state_degree(states, adj)["I"]

        state_prob = np.zeros((states.shape[0], self.num_states))
        state_prob[states == 0, 0] = (1 - beta) ** inf_deg[states == 0]
        state_prob[states == 0, 1] = 1 - (1 - beta) ** inf_deg[states == 0]
        state_prob[states == 1, 0] = alpha
        state_prob[states == 1, 1] = 1 - alpha
        return state_prob


class SIR(SingleEpidemics):
    """
        Class for  discrete SIR dynamics.

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

    def __init__(self, params):
        super(SIR, self).__init__(params, {"S": 0, "I": 1, "R": 2})

    def update(self, states=None, adj=None):
        if states is None:
            states = self.states
        if adj is None:
            adj = nx.to_numpy_array(self.graph)
        beta = self.params["infection"]
        alpha = self.params["recovery"]
        inf_deg = self.state_degree(states)["I"]
        inf_prob = 1 - (1 - beta) ** inf_deg
        rec_prob = alpha
        new_states = states * 1

        new_states[(states == 0) * (np.random.rand(*states.shape) < inf_prob)] = 1
        new_states[(states == 1) * (np.random.rand(*states.shape) < rec_prob)] = 2
        if np.sum(new_states == self.state_label["I"]) == 0:
            self.continue_simu = False
        self.states = new_states * 1
        return new_states

    def predict(self, states=None, adj=None):
        if states is None:
            states = self.states
        if adj is None:
            adj = nx.to_numpy_array(self.graph)

        beta = self.params["infection"]
        alpha = self.params["recovery"]
        inf_deg = self.state_degree(states, adj)["I"]

        state_prob = np.zeros((states.shape[0], self.num_states))
        state_prob[states == 0, 0] = (1 - beta) ** inf_deg[states == 0]
        state_prob[states == 0, 1] = 1 - (1 - beta) ** inf_deg[states == 0]
        state_prob[states == 0, 2] = 0
        state_prob[states == 1, 0] = 0
        state_prob[states == 1, 1] = 1 - alpha
        state_prob[states == 1, 2] = alpha
        state_prob[states == 2, 0] = 0
        state_prob[states == 2, 1] = 0
        state_prob[states == 2, 2] = 1

        return state_prob
