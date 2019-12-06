from .epidemics import *
import networkx as nx
import numpy as np


class SISSIS(DoubleEpidemics):
    def __init__(self, params):
        super(SISSIS, self).__init__(params, {"SS": 0, "IS": 1, "SI": 2, "II": 3})

    def update(self, states=None, adj=None):
        if states is None:
            states = self.states
        if adj is None:
            adj = nx.to_numpy_array(self.graph)
        state_deg = self.state_degree(states, adj)
        inf_prob = self.infection(states, state_deg)
        rec_prob = self.recovery(states, state_deg)
        status_to_g0 = np.zeros(states.shape)
        status_to_g1 = np.zeros(states.shape)

        status_to_g0[(states == 1) + (states == 3)] = 1
        status_to_g1[(states == 2) + (states == 3)] = 1

        # Events on node SS
        status_to_g0[(states == 0) * (np.random.rand(*states.shape) < inf_prob[0])] = 1
        status_to_g1[(states == 0) * (np.random.rand(*states.shape) < inf_prob[1])] = 1

        # Events on node IS
        status_to_g0[(states == 1) * (np.random.rand(*states.shape) < rec_prob[0])] = 0
        status_to_g1[(states == 1) * (np.random.rand(*states.shape) < inf_prob[1])] = 1

        # Events on node SI
        status_to_g0[(states == 2) * (np.random.rand(*states.shape) < inf_prob[0])] = 1
        status_to_g1[(states == 2) * (np.random.rand(*states.shape) < rec_prob[1])] = 0

        # Events on node II
        status_to_g0[(states == 3) * (np.random.rand(*states.shape) < rec_prob[0])] = 0
        status_to_g1[(states == 3) * (np.random.rand(*states.shape) < rec_prob[1])] = 0

        new_states = np.zeros(states.shape)

        new_states[(status_to_g0 == 0) * (status_to_g1 == 0)] = 0
        new_states[(status_to_g0 == 1) * (status_to_g1 == 0)] = 1
        new_states[(status_to_g0 == 0) * (status_to_g1 == 1)] = 2
        new_states[(status_to_g0 == 1) * (status_to_g1 == 1)] = 3

        if np.all(new_states == 0):
            continue_simu = False

        return new_states

    def predict(self, states=None, adj=None):
        if states is None:
            states = states
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
