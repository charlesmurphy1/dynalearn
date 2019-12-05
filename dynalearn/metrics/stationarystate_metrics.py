from .base_metrics import Metrics
import dynalearn as dl
import numpy as np
import tqdm
from scipy.optimize import bisect
import networkx as nx


class StationaryStateMetrics(Metrics):
    def __init__(
        self,
        parameters=None,
        num_samples=100,
        burn=10,
        reshuffle=0.1,
        tol=1e-2,
        verbose=1,
    ):
        self.parameters = parameters

        self.num_samples = num_samples
        self.burn = burn
        self.reshuffle = reshuffle
        self.tol = tol

        self._graph_model = None
        self._dynamics_model = None
        self._gnn_model = None

        super(StationaryStateMetrics, self).__init__(verbose)

    def compute(self, experiment):
        raise NotImplemented()

    def change_param(self, p):
        raise NotImplemented()

    def compute_stationary_states(self, model, x0, pb=None):
        adj = nx.to_numpy_array(self.graph_model.generate()[1])
        avg_x0 = self.avg(x0)
        samples = np.zeros((self.num_samples, self.dynamics_model.num_states))
        # samples = []
        it = 0
        ii = 0
        while ii < self.num_samples:
            it += 1
            x = model.update(x0, adj)
            avg_x = self.avg(x)
            dist = np.sqrt(((avg_x - avg_x0) ** 2).sum())
            avg_x0 = avg_x * 1
            if dist < self.tol and it > self.burn:
                samples[ii] = avg_x
                ii += 1
                if self.verbose and pb is not None:
                    pb.update()
                it = 0
                if np.random.rand() < self.reshuffle:
                    adj = nx.to_numpy_array(self.graph_model.generate()[1])

        return samples

    def avg(self, x):
        avg_x = []
        for i in range(self.dynamics_model.num_states):
            avg_x.append(np.mean(x == i))
        return np.array(avg_x)

    @property
    def graph_model(self):
        if self._graph_model is None:
            raise ValueError("No graph model is available")
        else:
            return self._graph_model

    @graph_model.setter
    def graph_model(self, graph_model):
        self._graph_model = graph_model

    @property
    def dynamics_model(self):
        if self._dynamics_model is None:
            raise ValueError("No dynamics model is available")
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
            raise ValueError("No GNN model is available")
        else:
            return self._gnn_model

    @gnn_model.setter
    def gnn_model(self, gnn_model):
        self._gnn_model = gnn_model


class EpidemicsSSMetrics(StationaryStateMetrics):
    def __init__(
        self,
        parameters=None,
        epsilon=1e-3,
        num_samples=100,
        burn=10,
        reshuffle=0.1,
        tol=1e-2,
        verbose=1,
    ):
        self.epsilon = epsilon
        super(EpidemicsSSMetrics, self).__init__(
            parameters, num_samples, burn, reshuffle, tol, verbose
        )

    def epidemic_state(self):
        self.dynamics_model.params["init"] = 1 - self.epsilon
        g = self.graph_model.generate()[1]
        return self.dynamics_model.initialize_states(g)

    def absoring_state(self):
        self.dynamics_model.params["init"] = self.epsilon
        g = self.graph_model.generate()[1]
        return self.dynamics_model.initialize_states(g)

    def compute(self, experiment):

        self.dynamics_model = experiment.dynamics_model
        self.gnn_model = experiment.model
        n_prev = self.gnn_model.num_nodes * 1

        self.gnn_model.num_nodes = self.graph_model.num_nodes

        if self.verbose:
            p_bar = tqdm.tqdm(
                range(4 * len(self.parameters) * self.num_samples),
                "Computing stationary states",
            )

        for p in self.parameters:
            self.change_param(p)
            low_ss = self.compute_stationary_states(
                self.dynamics_model, self.absoring_state(), p_bar
            )
            high_ss = self.compute_stationary_states(
                self.dynamics_model, self.epidemic_state(), p_bar
            )
            self.data[f"p = {p}/true_low_ss"] = low_ss
            self.data[f"p = {p}/true_high_ss"] = high_ss

        for p in self.parameters:
            self.change_param(p)
            low_ss = self.compute_stationary_states(
                self.gnn_model, self.absoring_state(), p_bar
            )
            high_ss = self.compute_stationary_states(
                self.gnn_model, self.epidemic_state(), p_bar
            )
            self.data[f"p = {p}/gnn_low_ss"] = low_ss
            self.data[f"p = {p}/gnn_high_ss"] = high_ss

        if self.verbose:
            p_bar.close()
        self.data[f"parameters"] = self.parameters


class PoissonEpidemicsSSMetrics(EpidemicsSSMetrics):
    def __init__(
        self,
        num_nodes=2000,
        parameters=None,
        num_k=3,
        epsilon=1e-3,
        num_samples=100,
        burn=10,
        reshuffle=0.1,
        tol=1e-2,
        verbose=1,
    ):
        self.num_nodes = num_nodes
        self.num_k = num_k
        if parameters is None:
            parameters = np.linspace(0.1, 10, 10)
        super(PoissonEpidemicsSSMetrics, self).__init__(
            parameters, epsilon, num_samples, burn, reshuffle, tol, verbose
        )
        self.change_param(parameters[0])

    def change_param(self, avgk):
        poisson_dist = dl.meanfields.poisson_distribution(avgk, num_k=self.num_k)
        self.graph_model = dl.graphs.DegreeSequenceGraph(
            {"N": self.num_nodes, "degree_dist": poisson_dist}
        )
