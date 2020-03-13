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
        self.burn = config.burn
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

    def get_samples(self, gen_x0=None, pb=None):

        self.model.graph = self.graph_model.generate()[1]
        if gen_x0 is None:
            gen_x0 = self.dynamics.initial_states
        samples = np.zeros((self.num_samples, self.model.num_nodes))

        for i in range(self.num_samples):
            self.model.graph = self.graph_model.generate()[1]
            samples[i] = self.burning(gen_x0(), self.burn)
            if self.verbose and pb is not None:
                pb.update()
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

    def get_initial_state(self, inf_fraction):
        self.dynamics.params["init"] = inf_fraction
        name, g = self.graph_model.generate()
        self.dynamics.graph = g
        x = self.dynamics.initial_states()
        return x

    def compute_stationary_states(self):
        avg = np.zeros((2, len(self.parameters), self.model.num_states))
        std = np.zeros((2, len(self.parameters), self.model.num_states))

        if self.verbose:
            pb = tqdm.tqdm(range(2 * len(self.parameters) * self.num_samples))
        else:
            pb = None

        inf_fraction = self.epsilon
        for i, p in enumerate(self.parameters):
            init_state = lambda: self.get_initial_state(inf_fraction)
            self.change_param(p)
            samples = self.get_samples(init_state, pb)
            avg_samples = self.avg(samples, axis=-1)
            avg[0, i] = np.mean(avg_samples, axis=-1)
            std[0, i] = np.std(avg_samples, axis=-1)
            inf_fraction = 1 - avg[0, i, 0]
            if inf_fraction < self.epsilon:
                inf_fraction = self.epsilon
            elif inf_fraction > 1 - self.epsilon:
                inf_fraction = 1 - self.epsilon

        inf_fraction = 1 - self.epsilon
        for i, p in enumerate(self.parameters[::-1]):
            i = -1 - i
            init_state = lambda: self.get_initial_state(inf_fraction)
            self.change_param(p)
            samples = self.get_samples(init_state, pb)
            avg_samples = self.avg(samples, axis=-1)
            avg[1, i] = np.mean(avg_samples, axis=-1)
            std[1, i] = np.std(avg_samples, axis=-1)
            inf_fraction = 1 - avg[1, i, 0]
            if inf_fraction < self.epsilon:
                inf_fraction = self.epsilon
            elif inf_fraction > 1 - self.epsilon:
                inf_fraction = 1 - self.epsilon

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
