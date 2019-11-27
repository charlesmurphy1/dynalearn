from dynalearn.dynamics import *
import networkx as nx
import numpy as np


class ComplexContagionSIS(SingleEpidemics):
    def __init__(self, activation_f, deactivation_f, init_state=None):
        super(ComplexContagionSIS, self).__init__({"S": 0, "I": 1}, init_state)
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


class ComplexContagionSIR(SingleEpidemics):
    def __init__(self, activation_f, deactivation_f, init_state=None):
        super(ComplexContagionSIR, self).__init__({"S": 0, "I": 1, "R": 2}, init_state)
        self.act_f = activation_f
        self.deact_f = deactivation_f

    def transition(self):
        inf_prob = self.act_f(self.state_degree(self.states))
        rec_prob = self.deact_f(self.state_degree(self.states))
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

        inf_prob = self.act_f(self.state_degree(states, adj))
        rec_prob = self.deact_f(self.state_degree(states, adj))

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


def sigmoid(x):
    return 1.0 / (np.exp(-x) + 1)


def soft_threshold_activation(state_degree, mu, beta):
    s_deg = np.array([state_degree[k] for k in state_degree]).T
    degree = np.sum(s_deg, axis=-1)
    act_prob = sigmoid(beta * (state_degree["I"] / degree - mu))
    act_prob[degree == 0] = 0
    return act_prob


class SoftThresholdSIS(ComplexContagionSIS):
    def __init__(self, mu, beta, recovery_prob, init_state=None):

        act_f = lambda l: soft_threshold_activation(l, mu, beta)
        deact_f = lambda l: recovery_prob * np.ones(l["S"].shape)

        super(SoftThresholdSIS, self).__init__(act_f, deact_f, init_state)
        self.params["mu"] = mu
        self.params["beta"] = beta
        self.params["recovery_prob"] = recovery_prob


class SoftThresholdSIR(ComplexContagionSIR):
    def __init__(self, mu, beta, recovery_prob, init_state=None):

        act_f = lambda l: soft_threshold_activation(l, mu, beta)
        deact_f = lambda l: recovery_prob * np.ones(l["S"].shape)
        super(SoftThresholdSIR, self).__init__(act_f, deact_f, init_state)

        self.params["mu"] = mu
        self.params["beta"] = beta
        self.params["recovery_prob"] = recovery_prob


def nonlinear_activation(state_degree, tau, alpha):
    act_prob = (1 - (1 - tau) ** state_degree["I"]) ** alpha
    # act_prob[degree == 0] = 0
    return act_prob


class NonLinearSIS(ComplexContagionSIS):
    def __init__(self, infection_prob, recovery_prob, alpha, init_state=None):

        act_f = lambda l: nonlinear_activation(l, infection_prob, alpha)
        deact_f = lambda l: recovery_prob * np.ones(l["S"].shape)

        super(NonLinearSIS, self).__init__(act_f, deact_f, init_state)
        self.params["infection_prob"] = infection_prob
        self.params["recovery_prob"] = recovery_prob
        self.params["alpha"] = alpha


class NonLinearSIR(ComplexContagionSIR):
    def __init__(self, infection_prob, recovery_prob, alpha, init_state=None):

        act_f = lambda l: nonlinear_activation(l, infection_prob, alpha)
        deact_f = lambda l: recovery_prob * np.ones(l["S"].shape)

        super(NonLinearSIR, self).__init__(act_f, deact_f, init_state)
        self.params["infection_prob"] = infection_prob
        self.params["recovery_prob"] = recovery_prob
        self.params["alpha"] = alpha
