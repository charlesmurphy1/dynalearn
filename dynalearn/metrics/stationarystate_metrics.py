from .base import Metrics
import dynalearn as dl
import numpy as np
import tqdm
from scipy.optimize import bisect
import networkx as nx
from abc import abstractmethod


class StationaryStateMetrics(Metrics):
    def __init__(self, config, verbose=1):
        self.__config = config
        self.parameters = config.ss_parameters
        self.num_samples = config.num_samples
        self.initial_burn = config.initial_burn
        self.burn = config.burn
        self.reshuffle = config.reshuffle
        self.tol = config.tol

        self._graph_model = None
        self._dynamics_model = None
        self._gnn_model = None

        super(StationaryStateMetrics, self).__init__(verbose)

    @abstractmethod
    def change_param(self, p):
        raise NotImplementedError("change_param must be implemented.")

    def compute_stationary_states(self, model, x0, pb=None):
        adj = nx.to_numpy_array(self.graph_model.generate()[1])
        x = x0 * 1
        avg_x0 = self.avg(x0)
        samples = np.zeros((self.num_samples, self.dynamics_model.num_states))
        # samples = []
        i = 0

        x = self.burning(x, adj, self.initial_burn)
        while i < self.num_samples:
            avg_x = self.avg(x)
            dist = np.sqrt(np.sum((avg_x - avg_x0) ** 2))
            avg_x0 = avg_x * 1
            if dist < self.tol:
                samples[i] = avg_x
                i += 1
                if self.verbose and pb is not None:
                    pb.update()
                if np.random.rand() < self.reshuffle:
                    adj = nx.to_numpy_array(self.graph_model.generate()[1])
            x = model.burning(x, adj, self.burn)

        return np.mean(samples, axis=0), np.std(samples, axis=0)

    def burning(self, x, adj, burn=1):
        for i in range(burn):
            x = model.sample(x, adj)
        return x

    def avg(self, x):
        avg_x = []
        for i in range(self.dynamics_model.num_states):
            avg_x.append(np.mean(x == i))
        return np.array(avg_x)

    @property
    def graph_model(self):
        if self._graph_model is None:
            raise ValueError("graph model is unavailable.")
        else:
            return self._graph_model

    @graph_model.setter
    def graph_model(self, graph_model):
        name = type(graph_model).__name__
        params = graph_model.params
        param_dict = {"name": name, "params": params}
        self._graph_model = dl.graphs.get(param_dict)

    @property
    def dynamics_model(self):
        if self._dynamics_model is None:
            raise ValueError("dynamics model is unavailable.")
        else:
            return self._dynamics_model

    @dynamics_model.setter
    def dynamics_model(self, dynamics_model):
        name = type(dynamics_model).__name__
        params = dynamics_model.params
        param_dict = {"name": name, "params": params}
        self._dynamics_model = dl.dynamics.get(param_dict)

    @property
    def gnn_model(self):
        if self._gnn_model is None:
            raise ValueError("GNN model is unavailable.")
        else:
            return self._gnn_model

    @gnn_model.setter
    def gnn_model(self, gnn_model):
        self._gnn_model = gnn_model


class EpidemicsSSMetrics(StationaryStateMetrics):
    def __init__(self, config, verbose=1):
        self.epsilon = config.epsilon
        super(EpidemicsSSMetrics, self).__init__(config, verbose)

    def epidemic_state(self):
        self.dynamics_model.params["init"] = 1 - self.epsilon
        g = self.graph_model.generate()[1]
        return self.dynamics_model.initialize_states(g)

    def absoring_state(self):
        self.dynamics_model.params["init"] = self.epsilon
        name, g = self.graph_model.generate()
        # print(name, type(g))
        return self.dynamics_model.initialize_states(g)

    def compute(self, experiment):

        self.dynamics_model = experiment.dynamics_model
        self.gnn_model = experiment.model
        n_prev = self.gnn_model.num_nodes * 1

        self.gnn_model.num_nodes = self.graph_model.num_nodes

        if self.verbose:
            p_bar = tqdm.tqdm(
                range(4 * len(self.parameters) * self.num_samples),
                "Computing " + self.__class__.__name__,
            )

        self.data["parameters"] = self.parameters

        true_low_avg_ss = np.zeros(
            (len(self.parameters), self.dynamics_model.num_states)
        )
        true_low_std_ss = np.zeros(
            (len(self.parameters), self.dynamics_model.num_states)
        )
        true_high_avg_ss = np.zeros(
            (len(self.parameters), self.dynamics_model.num_states)
        )
        true_high_std_ss = np.zeros(
            (len(self.parameters), self.dynamics_model.num_states)
        )

        for i, p in enumerate(self.parameters):
            self.change_param(p)
            low = self.compute_stationary_states(
                self.dynamics_model, self.absoring_state(), p_bar
            )
            high = self.compute_stationary_states(
                self.dynamics_model, self.epidemic_state(), p_bar
            )
            true_low_avg_ss[i], true_low_std_ss[i] = low[0], low[1]
            true_high_avg_ss[i], true_high_std_ss[i] = high[0], high[1]

        self.data["true_low_avg"] = true_low_avg_ss
        self.data["true_low_std"] = true_low_std_ss
        self.data["true_high_avg"] = true_high_avg_ss
        self.data["true_high_std"] = true_high_std_ss

        gnn_low_avg_ss = np.zeros(
            (len(self.parameters), self.dynamics_model.num_states)
        )
        gnn_low_std_ss = np.zeros(
            (len(self.parameters), self.dynamics_model.num_states)
        )
        gnn_high_avg_ss = np.zeros(
            (len(self.parameters), self.dynamics_model.num_states)
        )
        gnn_high_std_ss = np.zeros(
            (len(self.parameters), self.dynamics_model.num_states)
        )
        for i, pp in enumerate(self.parameters):
            self.change_param(pp)
            low = self.compute_stationary_states(
                self.gnn_model, self.absoring_state(), p_bar
            )
            high = self.compute_stationary_states(
                self.gnn_model, self.epidemic_state(), p_bar
            )
            gnn_low_avg_ss[i], gnn_low_std_ss[i] = low[0], low[1]
            gnn_high_avg_ss[i], gnn_high_std_ss[i] = high[0], high[1]

        self.data["gnn_low_avg"] = gnn_low_avg_ss
        self.data["gnn_low_std"] = gnn_low_std_ss
        self.data["gnn_high_avg"] = gnn_high_avg_ss
        self.data["gnn_high_std"] = gnn_high_std_ss

        if self.verbose:
            p_bar.close()


class PoissonEpidemicsSSMetrics(EpidemicsSSMetrics):
    def __init__(self, config, verbose=1):
        self.num_nodes = config.num_nodes
        self.num_k = config.num_k
        super(PoissonEpidemicsSSMetrics, self).__init__(config, verbose)
        self.change_param(config.ss_parameters[0])

    def change_param(self, avgk):
        poisson_dist = dl.utilities.poisson_distribution(avgk, self.num_k)
        self.graph_model = dl.graphs.DegreeSequenceGraph(
            {"N": self.num_nodes, "degree_dist": poisson_dist}
        )
