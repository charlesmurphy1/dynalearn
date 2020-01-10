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

    def sample(self, states=None, adj=None):
        if states is None:
            states = self.states
        else:
            self.states = states
            self.infected = set(np.where(states == np.state_label["I"])[0])

        if adj is None:
            g = self.graph
        else:
            g = nx.from_numpy_array(adj)

        beta = self.params["infection"]
        gamma = self.params["recovery"]

        N = g.number_of_nodes()
        new_infected = self.infected.copy()
        for i in self.infected:

            # Infection phase
            for j in g.neirhbors(i):
                if j not in self.infected and np.random.rand() < beta:
                    new_infected.add(j)

            # Recovery phase
            if np.random.rand() < gamma:
                new_infected.remove(i)

        self.states[set(range(N)) - new_infected] = self.state_label["S"]
        self.states[new_infected] = self.state_label["I"]
        self.infected = new_infected

        return self.states

    def predict(self, states=None, adj=None):
        if states is None:
            states = self.states
        if adj is None:
            adj = nx.to_numpy_array(self.graph)

        beta = self.params["infection"]
        gamma = self.params["recovery"]
        inf_deg = self.state_degree(states, adj)["I"]

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

    def sample(self, states=None, adj=None):
        if states is None:
            states = self.states
        else:
            self.states = states
            self.infected = set(np.where(states == np.state_label["I"])[0])
            self.recovered = set(np.where(states == np.state_label["R"])[0])

        if adj is None:
            g = self.graph
        else:
            g = nx.from_numpy_array(adj)

        beta = self.params["infection"]
        gamma = self.params["recovery"]

        N = g.number_of_nodes()
        new_infected = self.infected.copy()
        new_recovered = self.recovered.copy()
        for i in self.infected:

            # Infection phase
            for j in g.neirhbors(i):
                if (
                    j not in self.infected
                    and j not in self.recovered
                    and np.random.rand() < beta
                ):
                    new_infected.add(j)

            # Recovery phase
            if np.random.rand() < gamma:
                new_infected.remove(i)
                new_recovered.add(i)

        self.states[new_infected] = self.state_label["I"]
        self.states[new_recovered] = self.state_label["R"]
        self.infected = new_infected
        self.recovered = new_recovered

        return self.states

    def predict(self, states=None, adj=None):
        if states is None:
            states = self.states
        if adj is None:
            adj = nx.to_numpy_array(self.graph)

        beta = self.params["infection"]
        gamma = self.params["recovery"]
        inf_deg = self.state_degree(states, adj)["I"]

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
