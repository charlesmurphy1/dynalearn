from .base import *
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

    def sample(self, states):
        infected = set(np.where(states == self.state_label["I"])[0])

        beta = self.params["infection"]
        gamma = self.params["recovery"]

        N = self.graph.number_of_nodes()
        new_states = states.copy()
        for i in infected:

            # Infection phase
            for j in self.graph.neighbors(i):
                if j not in infected and np.random.rand() < beta:
                    new_states[j] = self.state_label["I"]

            # Recovery phase
            if np.random.rand() < gamma:
                new_states[i] = self.state_label["S"]

        return new_states

    def predict(self, states):

        beta = self.params["infection"]
        gamma = self.params["recovery"]
        inf_deg = self.state_degree(states)["I"]

        state_prob = np.zeros((states.shape[0], self.num_states))
        state_prob[states == 0, 0] = (1 - beta) ** inf_deg[states == 0]
        state_prob[states == 0, 1] = 1 - (1 - beta) ** inf_deg[states == 0]
        state_prob[states == 1, 0] = gamma
        state_prob[states == 1, 1] = 1 - gamma
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
        self.recovered = set()
        super(SIR, self).__init__(params, {"S": 0, "I": 1, "R": 2})

    def sample(self, states):
        infected = set(np.where(states == self.state_label["I"])[0])
        recovered = set(np.where(states == self.state_label["R"])[0])

        beta = self.params["infection"]
        gamma = self.params["recovery"]

        N = g.number_of_nodes()
        new_states = states.copy()
        for i in infected:
            # Infection phase
            for j in g.neighbors(i):
                if (
                    j not in self.infected
                    and j not in self.recovered
                    and np.random.rand() < beta
                ):
                    new_states[j] = self.state_label["I"]

            # Recovery phase
            if np.random.rand() < gamma:
                new_states[j] = self.state_label["R"]

        return new_states

    def predict(self, states):

        beta = self.params["infection"]
        gamma = self.params["recovery"]
        inf_deg = self.state_degree(states, self.adj)["I"]

        state_prob = np.zeros((states.shape[0], self.num_states))
        state_prob[states == 0, 0] = (1 - beta) ** inf_deg[states == 0]
        state_prob[states == 0, 1] = 1 - (1 - beta) ** inf_deg[states == 0]
        state_prob[states == 0, 2] = 0
        state_prob[states == 1, 0] = 0
        state_prob[states == 1, 1] = 1 - gamma
        state_prob[states == 1, 2] = gamma
        state_prob[states == 2, 0] = 0
        state_prob[states == 2, 1] = 0
        state_prob[states == 2, 2] = 1

        return state_prob
