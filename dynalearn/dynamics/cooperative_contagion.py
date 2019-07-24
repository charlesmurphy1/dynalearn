from dynalearn.dynamics import *
import networkx as nx
import numpy as np


class CooperativeSISDynamics(EpidemicDynamics):
    def __init__(self, infection, recovery, init_state=None):
        super(CooperativeSISDynamics, self).__init__(
            {"SS": 0, "IS": 1, "SI": 2, "II": 3}, init_state
        )
        self.infection = infection
        self.recovery = recovery

    def transition(self):
        state_deg = self.state_degree(self.states)
        inf_prob = self.infection(state_deg)
        rec_prob = self.recovery(state_deg)
        new_states = self.states * 1

        # Events on node SS
        new_states[
            (self.states == 0) * (np.random.rand(*self.states.shape) < inf_prob[0])
        ] = 1
        new_states[
            (self.states == 0) * (np.random.rand(*self.states.shape) < inf_prob[1])
        ] = 2

        # Events on node IS
        new_states[
            (self.states == 1) * (np.random.rand(*self.states.shape) < inf_prob[1])
        ] = 3
        new_states[
            (self.states == 1) * (np.random.rand(*self.states.shape) < rec_prob[0])
        ] = 0

        # Events on node SI
        new_states[
            (self.states == 2) * (np.random.rand(*self.states.shape) < inf_prob[0])
        ] = 3
        new_states[
            (self.states == 2) * (np.random.rand(*self.states.shape) < rec_prob[1])
        ] = 0

        # Events on node II
        new_states[
            (self.states == 3) * (np.random.rand(*self.states.shape) < rec_prob[0])
        ] = 2
        new_states[
            (self.states == 3) * (np.random.rand(*self.states.shape) < rec_prob[1])
        ] = 1

        if np.all(new_states == 0):
            continue_simu = False

        return new_states

    def predict(self, states, adj=None):
        state_deg = self.state_degree(self.states, adj)
        inf_prob = self.infection(states, state_deg)
        rec_prob = self.recovery(states, state_deg)

        state_prob = np.zeros((states.shape[0], self.num_states))

        # SS nodes
        state_prob[states == 0, 0] = 1 - (
            inf_prob[0][states == 0] + inf_prob[1][states == 0]
        )
        state_prob[states == 0, 1] = inf_prob[1][states == 0]
        state_prob[states == 0, 2] = inf_prob[0][states == 0]

        # IS nodes
        state_prob[states == 1, 0] = rec_prob[states == 1][0]
        state_prob[states == 1, 2] = 1 - rec_prob[states == 1][0]
        state_prob[states == 1, 3] = inf_prob[1][states == 1]

        # SI nodes
        state_prob[states == 2, 0] = rec_prob[states == 2][1]
        state_prob[states == 2, 1] = 1 - rec_prob[states == 2][1]
        state_prob[states == 2, 3] = inf_prob[0][states == 2]

        # II nodes
        state_prob[states == 3, 1] = rec_prob[states == 3][0]
        state_prob[states == 3, 2] = rec_prob[states == 3][1]
        state_prob[states == 3, 3] = 1 - (
            rec_prob[states == 3][0] + rec_prob[states == 3][1]
        )

        return state_prob


class SimpleCouplingCoopSISDynamics(object):
    def __init__(self, infection_prob, recovery_prob, coupling, init_state=None):
        self.infection_prob = infection_prob
        self.recovery_prob = recovery_prob
        self.coupling = coupling
        super(SimpleCouplingCoopSISDynamics, self).__init__()

    def infection(self, states, state_degrees):

        alpha = self.infection_prob
        beta = self.recovery_prob
        c = self.coupling
        inf0 = np.zeros(states.shape)
        inf1 = np.zeros(states.shape)

        # Node SS
        inf0[states == 0] = (
            1
            - (1 - alpha[0]) ** state_degrees["IS"][states == 0]
            * (1 - c * alpha[0]) ** state_degrees["II"][states == 0]
        )
        inf1[states == 0] = (
            1
            - (1 - alpha[1]) ** state_degrees["SI"][states == 0]
            * (1 - c * alpha[1]) ** state_degrees["II"][states == 0]
        )

        # Node IS
        inf1[states == 1] = (
            1
            - (1 - c * alpha[1]) ** state_degrees["SI"][states == 1]
            * (1 - c * alpha[1]) ** state_degrees["II"][states == 1]
        )

        # Node SI
        inf0[states == 2] = (
            1
            - (1 - c * alpha[0]) ** state_degrees["IS"][states == 2]
            * (1 - c * alpha[0]) ** state_degrees["II"][states == 2]
        )
        return inf0, inf1

    def recovery(self, states, state_degrees):

        alpha = self.infection_prob
        beta = self.recovery_prob
        c = self.coupling
        rec0 = np.ones(states.shape) * beta[0]
        rec1 = np.ones(states.shape) * beta[1]

        return inf0, inf1
