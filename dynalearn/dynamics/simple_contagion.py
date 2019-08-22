from dynalearn.dynamics import *
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

    def __init__(self, infection_prob, recovery_prob, init_state=None):
        super(SIS, self).__init__({"S": 0, "I": 1}, init_state)

        self.params["infection_prob"] = infection_prob
        self.params["recovery_prob"] = recovery_prob

    def transition(self):
        beta = self.params["infection_prob"]
        alpha = self.params["recovery_prob"]
        inf_deg = self.state_degree(self.states)["I"]
        inf_prob = 1 - (1 - beta) ** inf_deg
        rec_prob = alpha
        new_states = self.states * 1

        new_states[
            (self.states == 0) * (np.random.rand(*self.states.shape) < inf_prob)
        ] = 1
        new_states[
            (self.states == 1) * (np.random.rand(*self.states.shape) < rec_prob)
        ] = 0

        if np.sum(new_states == self.state_label["I"]) == 0:
            self.continue_simu = False

        return new_states

    def predict(self, states, adj=None):
        beta = self.params["infection_prob"]
        alpha = self.params["recovery_prob"]
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

    def __init__(self, infection_prob, recovery_prob, init_state=None):
        super(SIR, self).__init__({"S": 0, "I": 1, "R": 2}, init_state)

        self.params["infection_prob"] = infection_prob
        self.params["recovery_prob"] = recovery_prob

    def transition(self):
        beta = self.params["infection_prob"]
        alpha = self.params["recovery_prob"]
        inf_deg = self.state_degree(self.states)["I"]
        inf_prob = 1 - (1 - beta) ** inf_deg
        rec_prob = alpha
        new_states = self.states * 1

        new_states[
            (self.states == 0) * (np.random.rand(*self.states.shape) < inf_prob)
        ] = 1
        new_states[
            (self.states == 1) * (np.random.rand(*self.states.shape) < rec_prob)
        ] = 2
        if np.sum(new_states == self.state_label["I"]) == 0:
            self.continue_simu = False
        return new_states

    def predict(self, states, adj=None):

        beta = self.params["infection_prob"]
        alpha = self.params["recovery_prob"]
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
