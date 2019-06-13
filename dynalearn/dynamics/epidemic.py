"""

epidemic.py

Created by Charles Murphy on 26-06-18.
Copyright © 2018 Charles Murphy. All rights reserved.
Quebec, Canada

Defines DynamicalNetwork sub-classes for epidemic spreading dynamics.

"""

import numpy as np
from math import ceil
from dynalearn.dynamics import *


class EpidemicDynamics(Dynamics):
    def __init__(self, state_label, init_state):

        super(EpidemicDynamics, self).__init__(len(state_label))

        if "S" not in state_label or "I" not in state_label:
            raise ValueError("state_label must contain states 'S' and 'I'.")
        else:
            self.state_label = state_label
            self.inv_state_label = {state_label[i]: i for i in state_label}
        self.init_state = init_state

        for name, value in self.state_label.items():
            self.params["state_" + name] = value

    def initialize_states(self):
        N = self.graph.number_of_nodes()
        if self.init_state is not None:
            init_n_infected = ceil(N * self.init_state)
        else:
            init_n_infected = np.random.choice(range(N))
        nodeset = np.array(self.graph.nodes())
        ind = np.random.choice(nodeset, size=init_n_infected, replace=False)
        states = np.ones(N) * self.state_label["S"]
        states[ind] = self.state_label["I"]

        self.t = [0]

        self.continue_simu = True
        self.states = states

    def infected_degrees(self, states):
        N = self.graph.number_of_nodes()
        if len(states.shape) < 2:
            states = states.reshape(1, N)
        adj = nx.to_numpy_array(self.graph)
        cond = states == self.state_label["I"]
        return np.matmul(cond, adj)

    def get_avg_state(self):
        N = self.graph.number_of_nodes()
        state_dict = {l: np.zeros(N) for l in self.state_label}

        for v in self.graph.nodes():
            label = self.inv_state_label[self.states[v]]
            state_dict[label][v] = 1

        avg_states = {l: np.mean(state_dict[l]) for l in state_dict}
        std_states = {l: np.std(state_dict[l]) for l in state_dict}
        return avg_states, std_states

    def get_state_from_value(self, state):
        inv_state_label = {self.state_label[i]: i for i in self.state_label}

    def estimate_ltp(self, in_states, out_states):
        N = self.graph.number_of_nodes()
        adj = nx.to_numpy_array(self.graph)

        if len(in_states.shape) < 2:
            in_states = in_states.reshape(1, N)
            out_states = out_states.reshape(1, N)

        inf_deg = self.infected_degrees(in_states)

        avg_prob = {}
        var_prob = {}
        for in_s, in_l in self.state_label.items():
            for out_s, out_l in self.state_label.items():
                avg_prob[(in_s, out_s)] = np.zeros(N)
                var_prob[(in_s, out_s)] = np.zeros(N)
                for k in np.arange(np.max(inf_deg)):
                    avail_state = out_states[(in_states == in_l) * (inf_deg == k)]
                    n_sample = avail_state.shape[0]
                    out_condition = np.zeros(n_sample)
                    out_condition[avail_state == out_l] = 1
                    if n_sample > 0 and np.sum(out_condition) > 0:
                        avg_prob[(in_s, out_s)][int(k)] = np.mean(out_condition)
                        var_prob[(in_s, out_s)][int(k)] = np.std(
                            out_condition
                        ) / np.sqrt(n_sample)
        return avg_prob, var_prob

    def model_ltp(self, model, in_states):
        N = self.graph.number_of_nodes()
        adj = nx.to_numpy_array(self.graph)
        if len(in_states.shape) < 2:
            in_states = in_states.reshape(1, N)
            out_states = out_states.reshape(1, N)

        avg_prob = {}
        var_prob = {}

        inf_deg = self.infected_degrees(in_states)
        model_prediction = np.array(
            [model.predict([s, adj], steps=1) for s in in_states]
        )

        for in_s, in_l in self.state_label.items():
            for out_s, out_l in self.state_label.items():
                avg_prob[(in_s, out_s)] = np.zeros(N)
                var_prob[(in_s, out_s)] = np.zeros(N)
                for k in np.arange(int(np.max(inf_deg))):
                    avail_state = model_prediction[
                        (in_states == in_l) * (inf_deg == k), out_l
                    ]
                    n_sample = avail_state.shape[0]
                    if n_sample > 0:
                        avg_prob[(in_s, out_s)][int(k)] = np.mean(avail_state)
                        var_prob[(in_s, out_s)][int(k)] = np.std(avail_state) / np.sqrt(
                            n_sample
                        )

        return avg_prob, var_prob


class SISDynamics(EpidemicDynamics):
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
        super(SISDynamics, self).__init__({"S": 0, "I": 1}, init_state)

        self.params["infection_prob"] = infection_prob
        self.params["recovery_prob"] = recovery_prob

    def transition(self):
        beta = self.params["infection_prob"]
        alpha = self.params["recovery_prob"]
        inf_deg = np.squeeze(self.infected_degrees(self.states))
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

    def predict(self, states):
        N = self.graph.number_of_nodes()

        beta = self.params["infection_prob"]
        alpha = self.params["recovery_prob"]
        inf_deg = self.infected_degrees(states).squeeze()

        state_prob = np.zeros((states.shape[0], self.num_states))
        state_prob[states == 0, 0] = (1 - beta) ** inf_deg[states == 0]
        state_prob[states == 0, 1] = 1 - (1 - beta) ** inf_deg[states == 0]
        state_prob[states == 1, 0] = alpha
        state_prob[states == 1, 1] = 1 - alpha
        return state_prob

    def ltp(self, in_states):
        N = self.graph.number_of_nodes()
        p_inf = 1 - (1 - self.params["infection_prob"]) ** np.arange(N)
        p_rec = np.ones(N) * self.params["recovery_prob"]

        return {
            ("S", "S"): 1 - p_inf,
            ("S", "I"): p_inf,
            ("I", "S"): p_rec,
            ("I", "I"): 1 - p_rec,
        }


class SIRDynamics(EpidemicDynamics):
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
        super(SIRDynamics, self).__init__({"S": 0, "I": 1, "R": 2}, init_state)

        self.params["infection_prob"] = infection_prob
        self.params["recovery_prob"] = recovery_prob

    def transition(self):
        beta = self.params["infection_prob"]
        alpha = self.params["recovery_prob"]
        inf_deg = np.squeeze(self.infected_degrees(self.states))
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

    def predict(self, states):
        N = self.graph.number_of_nodes()

        beta = self.params["infection_prob"]
        alpha = self.params["recovery_prob"]
        inf_deg = self.infected_degrees(states).squeeze()

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

    def ltp(self, in_states):
        N = self.graph.number_of_nodes()
        p_inf = 1 - (1 - self.params["infection_prob"]) ** np.arange(N)
        p_rec = np.ones(N) * self.params["recovery_prob"]

        return {
            ("S", "S"): 1 - p_inf,
            ("S", "I"): p_inf,
            ("S", "R"): np.zeros(N),
            ("I", "S"): np.zeros(N),
            ("I", "I"): 1 - p_rec,
            ("I", "R"): p_rec,
            ("R", "S"): np.zeros(N),
            ("R", "I"): np.zeros(N),
            ("R", "R"): np.ones(N),
        }
