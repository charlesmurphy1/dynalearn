from .epidemics import *
import networkx as nx
import numpy as np
from scipy.special import lambertw


class ComplexContagionSIS(SingleEpidemics):
    def __init__(self, params, activation_f, deactivation_f):
        super(ComplexContagionSIS, self).__init__(params, {"S": 0, "I": 1})
        self.act_f = activation_f
        self.deact_f = deactivation_f

    def update(self, states=None, adj=None):
        if states is None:
            states = self.states
        if adj is None:
            adj = nx.to_numpy_array(self.graph)
        state_deg = self.state_degree(states)
        inf_prob = self.act_f(state_deg)
        rec_prob = self.deact_f(state_deg)
        new_states = states * 1

        new_states[(states == 0) * (np.random.rand(*states.shape) < inf_prob)] = 1
        new_states[(states == 1) * (np.random.rand(*states.shape) < rec_prob)] = 0

        if np.sum(new_states) == 0:
            continue_simu = False

        self.states = new_states
        return new_states

    def predict(self, states=None, adj=None):
        if states is None:
            states = self.states
        if adj is None:
            adj = nx.to_numpy_array(self.graph)
        inf_prob = self.act_f(self.state_degree(states, adj))
        rec_prob = self.deact_f(self.state_degree(states, adj))

        state_prob = np.zeros((states.shape[0], self.num_states))
        state_prob[states == 0, 0] = 1 - inf_prob[states == 0]
        state_prob[states == 0, 1] = inf_prob[states == 0]
        state_prob[states == 1, 0] = rec_prob[states == 1]
        state_prob[states == 1, 1] = 1 - rec_prob[states == 1]
        return state_prob


class ComplexContagionSIR(SingleEpidemics):
    def __init__(self, params, activation_f, deactivation_f):
        super(ComplexContagionSIR, self).__init__(params, {"S": 0, "I": 1, "R": 2})
        self.act_f = activation_f
        self.deact_f = deactivation_f

    def update(self, states=None, adj=None):
        if states is None:
            states = self.states
        if adj is None:
            adj = nx.to_numpy_array(self.graph)
        inf_prob = self.act_f(self.state_degree(states))
        rec_prob = self.deact_f(self.state_degree(states))
        new_states = states * 1

        new_states[(states == 0) * (np.random.rand(*states.shape) < inf_prob)] = 1
        new_states[(states == 1) * (np.random.rand(*states.shape) < rec_prob)] = 2

        if np.sum(new_states) == 0:
            continue_simu = False

        self.states = new_states

        return new_states

    def predict(self, states=None, adj=None):
        if states is None:
            states = self.states
        if adj is None:
            adj = nx.to_numpy_array(self.graph)
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
    def __init__(self, params):

        act_f = lambda l: soft_threshold_activation(
            l, params["threshold"], params["slope"]
        )
        deact_f = lambda l: params["recovery"] * np.ones(l["S"].shape)

        super(SoftThresholdSIS, self).__init__(params, act_f, deact_f)


class SoftThresholdSIR(ComplexContagionSIR):
    def __init__(self, params):

        act_f = lambda l: soft_threshold_activation(
            l, params["threshold"], params["slope"]
        )
        deact_f = lambda l: params["recovery"] * np.ones(l["S"].shape)

        super(SoftThresholdSIR, self).__init__(params, act_f, deact_f)


def nonlinear_activation(state_degree, tau, alpha):
    act_prob = (1 - (1 - tau) ** state_degree["I"]) ** alpha
    return act_prob


class NonLinearSIS(ComplexContagionSIS):
    def __init__(self, params):

        act_f = lambda l: nonlinear_activation(
            l, params["infection"], params["exponent"]
        )
        deact_f = lambda l: params["recovery"] * np.ones(l["S"].shape)

        super(NonLinearSIS, self).__init__(params, act_f, deact_f)


class NonLinearSIR(ComplexContagionSIR):
    def __init__(self, params):

        act_f = lambda l: nonlinear_activation(
            l, params["infection"], params["exponent"]
        )
        deact_f = lambda l: params["recovery"] * np.ones(l["S"].shape)

        super(NonLinearSIR, self).__init__(params, act_f, deact_f)


def sine_activation(state_degree, tau, epsilon, period):
    l = state_degree["I"]
    act_prob = (1 - (1 - tau) ** l) * (1 - epsilon * (np.sin(np.pi * l / period)) ** 2)
    return act_prob


class SineSIS(ComplexContagionSIS):
    def __init__(self, params):

        act_f = lambda l: sine_activation(
            l, params["infection"], params["amplitude"], params["period"]
        )
        deact_f = lambda l: params["recovery"] * np.ones(l["S"].shape)

        super(SineSIS, self).__init__(params, act_f, deact_f)


class SineSIR(ComplexContagionSIR):
    def __init__(self, params):

        act_f = lambda l: sine_activation(
            l, params["infection"], params["amplitude"], params["period"]
        )
        deact_f = lambda l: params["recovery"] * np.ones(l["S"].shape)

        super(SineSIR, self).__init__(params, act_f, deact_f)


def planck_activation(state_degree, temperature):
    l = state_degree["I"]
    gamma = (lambertw(-3 * np.exp(-3)) + 3).real
    Z = gamma ** 3 * temperature ** 3 / (np.exp(gamma) - 1)
    act_prob = l ** 3 / (np.exp(l / temperature) - 1) / Z
    return act_prob


class PlanckSIS(ComplexContagionSIS):
    def __init__(self, params):

        act_f = lambda l: planck_activation(l, params["temperature"])
        deact_f = lambda l: params["recovery"] * np.ones(l["S"].shape)

        super(PlanckSIS, self).__init__(params, act_f, deact_f)


class PlanckSIR(ComplexContagionSIR):
    def __init__(self, params):

        act_f = lambda l: planck_activation(l, params["temperature"])
        deact_f = lambda l: params["recovery"] * np.ones(l["S"].shape)

        super(PlanckSIR, self).__init__(params, act_f, deact_f)
