from .base import Metrics
import dynalearn as dl
import numpy as np
import tqdm
from scipy.optimize import bisect
import networkx as nx
from abc import abstractmethod


class StationaryStateMetrics(Metrics):
    def __init__(self, config, verbose=0):
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

    @abstractmethod
    def compute_stationary_states(self, model):
        raise NotImplementedError("compute_stationary_states must be implemented.")

    def compute(self, experiment):
        true_model = experiment.dynamics_model
        gnn_model = experiment.model

        if self.verbose:
            print("Computing " + self.__class__.__name__ + ": True")
        avg, std = self.compute_stationary_states(true_model)
        self.data[f"true_ss_avg"] = avg
        self.data[f"true_ss_std"] = std
        if self.verbose:
            print("Computing " + self.__class__.__name__ + ": GNN")
        avg, std = self.compute_stationary_states(gnn_model)
        self.data[f"gnn_ss_avg"] = avg
        self.data[f"gnn_ss_std"] = std

    def get_samples(self, model, x0, pb=None):

        model.graph = self.graph_model.generate()[1]
        x = x0 * 1
        samples = np.zeros((self.num_samples, model.graph.number_of_nodes()))
        x = self.burning(model, x, self.initial_burn)

        for i in range(self.num_samples):
            samples[i] = x
            if self.verbose and pb is not None:
                pb.update()

            if (i + 1) % self.reshuffle == 0:
                model.graph = self.graph_model.generate()[1]
                x = self.burning(model, x, self.initial_burn)

            x = self.burning(model, x, self.burn)
        return samples

    def burning(self, model, x, burn=1):

        for b in range(burn):
            x = model.sample(x)

        return x

    def avg(self, x):
        avg_x = np.zeros(self.dynamics_model.num_states)
        for i in range(self.dynamics_model.num_states):
            avg_x[i] = np.mean(x == i)
        return avg_x

    def std(self, x):
        std_x = np.zeros(self.dynamics_model.num_states)
        for i in range(self.dynamics_model.num_states):
            std_x[i] = np.std(x == i)
        return np.array(std_x)

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
    def __init__(self, config, verbose=0):
        self.epsilon = config.epsilon
        super(EpidemicsSSMetrics, self).__init__(config, verbose)

    def epidemic_state(self):
        self.dynamics_model.params["init"] = 1 - self.epsilon
        name, g = self.graph_model.generate()
        return self.dynamics_model.initial_states(g)

    def absoring_state(self):
        self.dynamics_model.params["init"] = self.epsilon
        name, g = self.graph_model.generate()
        return self.dynamics_model.initial_states(g)

    def compute_stationary_states(self, model):
        avg = np.zeros((2, len(self.parameters), self.dynamics_model.num_states))
        std = np.zeros((2, len(self.parameters), self.dynamics_model.num_states))

        if self.verbose:
            pb = tqdm.tqdm(range(2 * len(self.parameters) * self.num_samples))
        else:
            pb = None

        x0 = self.absoring_state()
        for i, p in enumerate(self.parameters):
            self.change_param(p)
            samples = self.get_samples(model, x0, pb)
            x0 = samples[-1]
            if self.dynamics_model.is_dead(x0):
                x0 = self.absoring_state()
            avg[0, i] = self.avg(samples)
            std[0, i] = self.std(samples)

        x0 = self.epidemic_state()
        for i, p in reversed(list(enumerate(self.parameters))):
            self.change_param(p)
            samples = self.get_samples(model, x0, pb)
            x0 = samples[-1]
            if self.dynamics_model.is_dead(x0):
                x0 = self.absoring_state()
            avg[1, i] = self.avg(samples)
            std[1, i] = self.std(samples)

        if self.verbose:
            pb.close()

        return avg, std


class PoissonEpidemicsSSMetrics(EpidemicsSSMetrics):
    def __init__(self, config, verbose=0):
        self.num_nodes = config.num_nodes
        self.num_k = config.num_k
        super(PoissonEpidemicsSSMetrics, self).__init__(config, verbose)
        self.change_param(config.ss_parameters[0])

    def change_param(self, avgk):
        poisson_dist = dl.utilities.poisson_distribution(avgk, self.num_k)
        self.graph_model = dl.graphs.DegreeSequenceGraph(
            {"N": self.num_nodes, "degree_dist": poisson_dist}
        )
