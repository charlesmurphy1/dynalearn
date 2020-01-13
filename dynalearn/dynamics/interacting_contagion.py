from .base import *
import networkx as nx
import numpy as np


class SISSIS(DoubleEpidemics):
    def __init__(self, params):
        super(SISSIS, self).__init__(params, {"SS": 0, "IS": 1, "SI": 2, "II": 3})

    # def sample(self, states):
    #
    #     p0, p1 = self.infection(states)
    #     q0, q1 = self.recovery(states)
    #
    #     ind_ss = np.where(states == self.state_label["SS"])[0]
    #     ind_is = np.where(states == self.state_label["IS"])[0]
    #     ind_si = np.where(states == self.state_label["SI"])[0]
    #     ind_ii = np.where(states == self.state_label["II"])[0]
    #
    #     N = self.graph.number_of_nodes()
    #     new_states = states.copy()
    #
    #     cond0 = np.random.rand(len(ind_ss)) < p0[ind_ss]
    #     cond1 = np.random.rand(len(ind_ss)) < p1[ind_ss]
    #     new_states[ind_ss][np.where(cond0 * ~cond1)[0]] = self.state_label["IS"]
    #     new_states[ind_ss][np.where(~cond0 * cond1)[0]] = self.state_label["SI"]
    #     new_states[ind_ss][np.where(cond0 * cond1)[0]] = self.state_label["II"]
    #
    #     cond1 = np.random.rand(len(ind_is)) < q0[ind_is]
    #     cond0 = np.random.rand(len(ind_is)) < p1[ind_is]
    #     new_states[ind_is][np.where(cond0 * ~cond1)[0]] = self.state_label["SS"]
    #     new_states[ind_is][np.where(~cond0 * cond1)[0]] = self.state_label["II"]
    #     new_states[ind_is][np.where(cond0 * cond1)[0]] = self.state_label["SI"]
    #
    #     cond1 = np.random.rand(len(ind_si)) < q0[ind_si]
    #     cond0 = np.random.rand(len(ind_si)) < p1[ind_si]
    #     new_states[ind_si][np.where(cond0 * ~cond1)[0]] = self.state_label["SS"]
    #     new_states[ind_si][np.where(~cond0 * cond1)[0]] = self.state_label["II"]
    #     new_states[ind_si][np.where(cond0 * cond1)[0]] = self.state_label["SI"]
    #
    #     cond1 = np.random.rand(len(ind_ii)) < q0[ind_ii]
    #     cond0 = np.random.rand(len(ind_ii)) < q1[ind_ii]
    #     new_states[ind_ii][np.where(cond0 * ~cond1)[0]] = self.state_label["SI"]
    #     new_states[ind_ii][np.where(~cond0 * cond1)[0]] = self.state_label["IS"]
    #     new_states[ind_ii][np.where(cond0 * cond1)[0]] = self.state_label["SS"]
    #
    #     return new_states

    def predict(self, states):
        state_deg = self.state_degree(states)
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
