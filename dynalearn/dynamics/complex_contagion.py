from dynalearn.dynamics import *
import networkx as nx
import numpy as np


class ComplexContagionSISDynamics(EpidemicDynamics):
    def __init__(self, activation_f, deactivation_f, init_state=None):
        super(ComplexContagionSISDynamics, self).__init__({"S": 0, "I": 1}, init_state)
        self.act_f = activation_f
        self.deact_f = deactivation_f

    def transition(self):
        state_deg = self.state_degree(self.states)
        inf_prob = self.act_f(state_deg)
        rec_prob = self.deact_f(state_deg)
        new_states = self.states * 1

        new_states[
            (self.states == 0) * (np.random.rand(*self.states.shape) < inf_prob)
        ] = 1
        new_states[
            (self.states == 1) * (np.random.rand(*self.states.shape) < rec_prob)
        ] = 0

        if np.sum(new_states) == 0:
            continue_simu = False

        return new_states

    def predict(self, states, adj=None):
        inf_prob = self.act_f(self.state_degree(states, adj))
        rec_prob = self.deact_f(self.state_degree(states, adj))

        state_prob = np.zeros((states.shape[0], self.num_states))
        state_prob[states == 0, 0] = 1 - inf_prob[states == 0]
        state_prob[states == 0, 1] = inf_prob[states == 0]
        state_prob[states == 1, 0] = rec_prob[states == 1]
        state_prob[states == 1, 1] = 1 - rec_prob[states == 1]

        return state_prob


class ComplexContagionSIRDynamics(EpidemicDynamics):
    def __init__(self, activation_f, deactivation_f, init_state=None):
        super(ComplexContagionSIRDynamics, self).__init__(
            {"S": 0, "I": 1, "R": 2}, init_state
        )
        self.act_f = activation_f
        self.deact_f = deactivation_f

    def transition(self):
        inf_prob = self.act_f(self.state_degree(self.states)).squeeze()
        rec_prob = self.deact_f(self.state_degree(self.states)).squeeze()
        new_states = self.states * 1

        new_states[
            (self.states == 0) * (np.random.rand(*self.states.shape) < inf_prob)
        ] = 1
        new_states[
            (self.states == 1) * (np.random.rand(*self.states.shape) < rec_prob)
        ] = 2

        if np.sum(new_states) == 0:
            continue_simu = False

        return new_states

    def predict(self, states, adj=None):

        inf_prob = self.act_f(self.state_degree(states, adj)).squeeze()
        rec_prob = self.deact_f(self.state_degree(states, adj)).squeeze()

        state_prob = np.zeros((states.shape[0], self.num_states))
        # Susceptible node
        state_prob[states == 0, 0] = 1 - inf_prob[states == 0]
        state_prob[states == 0, 1] = inf_prob[states == 0]
        state_prob[states == 0, 2] = 0
        # Infected node
        state_prob[states == 1, 0] = 0
        state_prob[states == 1, 1] = 1 - rec_prob[states == 1]
        state_prob[states == 1, 2] = rec_prob[states == 1]
        # Recovered node
        state_prob[states == 2, 0] = 0
        state_prob[states == 2, 1] = 0
        state_prob[states == 2, 2] = 1

        return state_prob


def soft_threshold_activation(state_degree, mu, beta):
    degree = np.sum([state_degree[k] for k in state_degree])
    act_prob = np.exp(beta * (state_degree["I"] / degree - mu) + 1).squeeze()
    act_prob[degree == 0] = 0
    return act_prob


class SoftThresholdSISDynamics(ComplexContagionSISDynamics):
    def __init__(self, mu, beta, recovery_prob, init_state=None):

        act_f = lambda l: soft_threshold_activation(l, mu, beta)
        deact_f = lambda l: recovery_prob * np.ones(l["S"].shape).squeeze()

        super(SoftThresholdSISDynamics, self).__init__(act_f, deact_f, init_state)
        self.params["mu"] = mu
        self.params["beta"] = beta
        self.params["recovery_prob"] = recovery_prob


class SoftThresholdSIRDynamics(ComplexContagionSIRDynamics):
    def __init__(self, mu, beta, recovery_prob, init_state=None):

        act_f = lambda l: soft_threshold_activation(l, mu, beta)
        deact_f = lambda l: recovery_prob * np.ones(l["S"].shape).squeeze()
        super(SoftThresholdSIRDynamics, self).__init__(act_f, deact_f, init_state)

        self.params["mu"] = mu
        self.params["beta"] = beta
        self.params["recovery_prob"] = recovery_prob
