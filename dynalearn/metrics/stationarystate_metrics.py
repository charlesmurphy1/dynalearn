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

        self.graph = None
        self.model = None
        self.dynamics = None

        super(StationaryStateMetrics, self).__init__(verbose)

    @abstractmethod
    def change_param(self, new_param):
        raise NotImplementedError("change_param must be implemented.")

    @abstractmethod
    def compute_stationary_states(self, model):
        raise NotImplementedError("compute_stationary_states must be implemented.")

    @abstractmethod
    def get_model(self, experiment):
        raise NotImplementedError("get_model must be implemented.")

    def compute(self, experiment):
        self.get_model(experiment)
        self.dynamics = experiment.dynamics_model

        if self.verbose:
            print("Computing " + self.__class__.__name__)
        avg, std = self.compute_stationary_states()
        self.data["parameters"] = self.parameters
        self.data["avg"] = avg
        self.data["std"] = std

    def get_samples(self, x0, pb=None):

        self.model.graph = self.graph_model.generate()[1]
        x = x0 * 1
        samples = np.zeros((self.num_samples, self.model.num_nodes))
        x = self.burning(x, self.initial_burn)

        for i in range(self.num_samples):
            samples[i] = x
            if self.dynamics.is_dead(x):
                x = self.burning(x, self.initial_burn)
                x = x0 * 1
            if self.verbose and pb is not None:
                pb.update()

            if (i + 1) % self.reshuffle == 0:
                self.model.graph = self.graph_model.generate()[1]
                x = self.burning(x, self.initial_burn)

            x = self.burning(x, self.burn)
        return samples

    def burning(self, x, burn=1):
        for b in range(burn):
            x = self.model.sample(x)
        return x

    def avg(self, x, axis=None):
        avg_x = []
        for i in range(self.model.num_states):
            avg_x.append(np.mean(x == i, axis=axis))
        return np.array(avg_x)


class EpidemicSSMetrics(StationaryStateMetrics):
    def __init__(self, config, verbose=0):
        self.epsilon = config.ss_epsilon
        super(EpidemicSSMetrics, self).__init__(config, verbose)

    def epidemic_state(self):
        self.dynamics.params["init"] = 1 - self.epsilon
        name, g = self.graph_model.generate()
        self.dynamics.graph = g
        return self.dynamics.initial_states()

    def absoring_state(self):
        self.dynamics.params["init"] = self.epsilon
        name, g = self.graph_model.generate()
        self.dynamics.graph = g
        return self.dynamics.initial_states()

    def compute_stationary_states(self):
        avg = np.zeros((2, len(self.parameters), self.model.num_states))
        std = np.zeros((2, len(self.parameters), self.model.num_states))

        if self.verbose:
            pb = tqdm.tqdm(range(2 * len(self.parameters) * self.num_samples))
        else:
            pb = None

        x0 = self.absoring_state()
        for i, p in enumerate(self.parameters):
            self.change_param(p)
            samples = self.get_samples(x0, pb)
            x0 = samples[-1]
            avg_samples = self.avg(samples, axis=-1)
            if self.dynamics.is_dead(x0):
                x0 = self.absoring_state()
            avg[0, i] = np.mean(avg_samples, axis=-1)
            std[0, i] = np.std(avg_samples, axis=-1)

        x0 = self.epidemic_state()
        for i, p in reversed(list(enumerate(self.parameters))):
            self.change_param(p)
            samples = self.get_samples(x0, pb)
            x0 = samples[-1]
            avg_samples = self.avg(samples, axis=-1)
            if self.dynamics.is_dead(x0):
                x0 = self.absoring_state()
            avg[1, i] = np.mean(avg_samples, axis=-1)
            std[1, i] = np.std(avg_samples, axis=-1)

        if self.verbose:
            pb.close()

        return avg, std


class PoissonESSMetrics(EpidemicSSMetrics):
    def __init__(self, config, verbose=0):
        self.num_nodes = config.num_nodes
        self.num_k = config.num_k
        super(PoissonESSMetrics, self).__init__(config, verbose)
        self.change_param(config.ss_parameters[0])

    def change_param(self, avgk):
        poisson_dist = dl.utilities.poisson_distribution(avgk, self.num_k)
        self.graph_model = dl.graphs.DegreeSequenceGraph(
            {"N": self.num_nodes, "degree_dist": poisson_dist}
        )


class TruePESSMetrics(PoissonESSMetrics):
    def __init__(self, config, verbose=0):
        super(TruePESSMetrics, self).__init__(config, verbose)

    def get_model(self, experiment):
        self.model = experiment.dynamics_model


class GNNPESSMetrics(PoissonESSMetrics):
    def __init__(self, config, verbose=0):
        super(GNNPESSMetrics, self).__init__(config, verbose)

    def get_model(self, experiment):
        self.model = experiment.model
