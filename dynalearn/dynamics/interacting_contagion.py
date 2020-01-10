from .base import *
import networkx as nx
import numpy as np


class SISSIS(DoubleEpidemics):
    def __init__(self, params):
        super(SISSIS, self).__init__(params, {"SS": 0, "IS": 1, "SI": 2, "II": 3})

    def sample(self, states=None, adj=None):
        if states is not None:
            self.states = states
            self.infected_1 = set(np.where(states == np.state_label["IS"])[0]) + set(
                np.where(states == np.state_label["II"])[0]
            )
            self.infected_2 = set(np.where(states == np.state_label["SI"])[0]) + set(
                np.where(states == np.state_label["II"])[0]
            )

        if adj is None:
            g = self.graph
        else:
            g = nx.from_numpy_array(adj)

        N = g.number_of_nodes()
        new_infected_1 = self.new_infected_1
        new_infected_2 = self.new_infected_2
        new_states = states.copy()

        for i in self.infected_1:
            # Infection phase of disease 1
            for j in g.neirhbors(i):
                prob = self.params["infection1"]
                if j in self.infected_2 or i in self.infected_2:
                    prob *= self.params["coupling"]
                if j not in self.infected_1 and np.random.rand() < prob:
                    new_infected_1.add(j)

            # Recovery phase of disease 1
            if np.random.rand() < self.params["recovery1"]:
                new_infected_1.remove(i)

        for i in self.infected_2:
            # Infection phase of disease 2
            for j in g.neirhbors(i):
                prob = self.params["infection2"]
                if j in self.infected_1 or i in self.infected_1:
                    prob *= self.params["coupling"]
                if j not in self.infected_2 and np.random.rand() < prob:
                    new_infected.add(j)

            # Recovery phase of disease 2
            if np.random.rand() < self.params["recovery2"]:
                new_infected_2.remove(i)

        self.states[set(range(N)) - new_infected_1 - new_infected_2] = self.state_label[
            "SS"
        ]
        self.states[new_infected_1] = self.state_label["IS"]
        self.states[new_infected_2] = self.state_label["SI"]
        self.states[np.intersect1d(new_infected_1, new_infected_2)] = self.state_label[
            "II"
        ]
        self.infected_1 = infected_1
        self.infected_2 = infected_2

        return self.states

    def predict(self, states=None, adj=None):
        if states is None:
            states = self.states
        if adj is None:
            adj = nx.to_numpy_array(self.graph)
        state_deg = self.state_degree(states, adj)
        p0, p1 = self.infection(states, state_deg)
        q0, q1 = self.recovery(states, state_deg)

        state_prob = np.zeros((states.shape[0], self.num_states))

        # SS nodes
        state_prob[states == 0, 0] = (1 - p0[states == 0]) * (1 - p1[states == 0])
        state_prob[states == 0, 1] = p0[states == 0] * (1 - p1[states == 0])
        state_prob[states == 0, 2] = (1 - p0[states == 0]) * p1[states == 0]
        state_prob[states == 0, 3] = p0[states == 0] * p1[states == 0]

        # IS nodes
        state_prob[states == 1, 0] = q0[states == 1] * (1 - p1[states == 1])
        state_prob[states == 1, 1] = (1 - q0[states == 1]) * (1 - p1[states == 1])
        state_prob[states == 1, 2] = q0[states == 1] * p1[states == 1]
        state_prob[states == 1, 3] = (1 - q0[states == 1]) * p1[states == 1]

        # SI nodes
        state_prob[states == 2, 0] = (1 - p0[states == 2]) * q1[states == 2]
        state_prob[states == 2, 1] = p0[states == 2] * q1[states == 2]
        state_prob[states == 2, 2] = (1 - p0[states == 2]) * (1 - q1[states == 2])
        state_prob[states == 2, 3] = p0[states == 2] * (1 - q1[states == 2])

        # II nodes
        state_prob[states == 3, 0] = q0[states == 3] * q1[states == 3]
        state_prob[states == 3, 1] = (1 - q0[states == 3]) * q1[states == 3]
        state_prob[states == 3, 2] = q0[states == 3] * (1 - q1[states == 3])
        state_prob[states == 3, 3] = (1 - q0[states == 3]) * (1 - q1[states == 3])

        return state_prob

    def infection(self, states, neighbor_states):

        alpha1 = self.params["infection1"]
        alpha2 = self.params["infection2"]
        c = self.params["coupling"]
        inf0 = np.zeros(states.shape)
        inf1 = np.zeros(states.shape)

        # Node SS
        inf0[states == 0] = (
            1
            - (1 - alpha1) ** neighbor_states["IS"][states == 0]
            * (1 - c * alpha1) ** neighbor_states["II"][states == 0]
        )
        inf1[states == 0] = (
            1
            - (1 - alpha2) ** neighbor_states["SI"][states == 0]
            * (1 - c * alpha2) ** neighbor_states["II"][states == 0]
        )

        # Node IS
        inf1[states == 1] = 1 - (1 - c * alpha2) ** (
            neighbor_states["SI"][states == 1] + neighbor_states["II"][states == 1]
        )

        # Node SI
        inf0[states == 2] = 1 - (1 - c * alpha1) ** (
            neighbor_states["IS"][states == 2] + neighbor_states["II"][states == 2]
        )
        return inf0, inf1

    def recovery(self, states, neighbor_states):
        rec0 = np.ones(states.shape) * self.params["recovery1"]
        rec1 = np.ones(states.shape) * self.params["recovery2"]

        return rec0, rec1
