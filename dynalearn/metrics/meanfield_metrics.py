from .base import Metrics
import dynalearn as dl
import numpy as np
import tqdm
from abc import abstractmethod
from scipy.optimize import approx_fprime
import time


def bisection(f, a, b, num_iter=100, p_bar=None):

    if f(a) * f(b) >= 0:
        print("Warning: f(a) and f(b) have the same sign")
        return (a + b) / 2

    for i in range(num_iter):
        x = (a + b) / 2
        if f(x) > 0:
            b = x
        else:
            a = x
        if p_bar is not None:
            p_bar.update()

    return x


class MeanfieldMetrics(Metrics):
    def __init__(self, config, verbose=0):
        self.__config = config
        self.parameters = config.mf_parameters
        self.p_range = config.p_range
        self.fp_finder = config.fp_finder
        self.model = None
        super(MeanfieldMetrics, self).__init__(verbose)

    def compute(self, experiment):
        self.get_model(experiment)
        if self.verbose != 0:
            print("Computing " + self.__class__.__name__)
        self.data["parameters"] = self.parameters
        self.data["thresholds"] = self.compute_thresholds()
        self.data["fixed_points"] = self.compute_fixed_points()

    @abstractmethod
    def compute_fixed_points(self, mf):
        raise NotImplementedError("compute_fixed_points must be implemented.")

    @abstractmethod
    def compute_thresholds(self, mf):
        raise NotImplementedError("compute_thresholds must be implemented.")

    @abstractmethod
    def change_param(self, mf, value):
        raise NotImplementedError("change_param must be implemented.")

    @abstractmethod
    def get_model(self, experiment):
        raise NotImplementedError("get_model must be implemented.")


class EpidemicsMFMetrics(MeanfieldMetrics):
    def __init__(self, config, verbose=0):
        self.epsilon = config.epsilon
        self.tol = config.tol
        self.fp_finder = config.fp_finder
        if config.discontinuous:
            self.criterion = self.__discontinuous_threshold_criterion
        else:
            self.criterion = self.__continuous_threshold_criterion

        super(EpidemicsMFMetrics, self).__init__(config, verbose)

    def compute_fixed_points(self):

        size = (len(self.parameters), self.model.num_states)

        if self.verbose == 1:
            p_bar = tqdm.tqdm(range(2 * len(self.parameters)), "Fixed points")

        x0 = self.absorbing_state()
        low_fp = np.zeros(size)
        for i, p in enumerate(self.parameters):
            self.change_param(p)
            x = self.model.search_fixed_point(x0=x0, fp_finder=self.fp_finder)
            low_fp[i] = self.model.to_avg(x)
            if self.verbose == 1:
                p_bar.update()

        high_fp = np.zeros(size)
        x0 = self.epidemic_state()
        for i, p in reversed(list(enumerate(self.parameters))):
            self.change_param(p)
            x = self.model.search_fixed_point(x0=x0, fp_finder=self.fp_finder)
            high_fp[i] = self.model.to_avg(x)
            if self.verbose == 1:
                p_bar.update()
        if self.verbose == 1:
            p_bar.close()

        return np.concatenate(
            (low_fp.reshape(1, *size), high_fp.reshape(1, *size)), axis=0
        )

    def compute_thresholds(self):
        low_f = lambda p: self.criterion(self.absorbing_state().reshape(-1), p)
        high_f = lambda p: self.criterion(self.epidemic_state().reshape(-1), p)

        thresholds = np.array([])
        p_min = self.p_range[0]
        p_max = self.p_range[1]

        num_iter = np.ceil(np.log2(p_max - p_min) - np.log2(self.tol)).astype("int")
        if self.verbose == 1:
            p_bar = tqdm.tqdm(range(2 * num_iter), "Thresholds")
        else:
            p_bar = None

        if low_f(p_min) * low_f(p_max) < 0:
            low_threshold = bisection(
                low_f, p_min, p_max, num_iter=num_iter, p_bar=p_bar
            )
            thresholds = np.append(thresholds, low_threshold)

        if high_f(p_min) * high_f(p_max) < 0:
            high_threshold = bisection(
                high_f, p_min, p_max, num_iter=num_iter, p_bar=p_bar
            )
            thresholds = np.append(thresholds, high_threshold)

        if self.verbose == 1:
            p_bar.close()
        return thresholds

    def epidemic_state(self):
        x = np.ones(self.model.array_shape).astype(self.model.dtype) * self.epsilon
        x[0] = 1 - self.epsilon
        x = 1 - x
        return self.model.normalize_state(x)

    def absorbing_state(self):
        x = np.ones(self.model.array_shape) * self.epsilon
        x[0] = 1 - self.epsilon
        return self.model.normalize_state(x)

    def __continuous_threshold_criterion(self, states, p):
        self.change_param(p)
        val = self.model.stability(states) - 1
        return val

    def __discontinuous_threshold_criterion(self, states, p):
        self.change_param(p)
        fp = self.model.search_fixed_point(x0=states, fp_finder=self.fp_finder)
        s = self.model.to_avg(fp)[0]
        if s >= 1 - 1e-2:
            val = -1
        else:
            val = 1
        return val


class PoissonEMFMetrics(EpidemicsMFMetrics):
    def __init__(self, config, verbose=0):
        self.num_k = config.num_k
        super(PoissonEMFMetrics, self).__init__(config, verbose)

    def change_param(self, avgk):
        degree_dist = dl.utilities.poisson_distribution(avgk, self.num_k)
        self.model.degree_dist = degree_dist


class TruePEMFMetrics(PoissonEMFMetrics):
    def __init__(self, config, verbose=0):
        super(TruePEMFMetrics, self).__init__(config, verbose)

    def get_model(self, experiment):
        degree_dist = dl.utilities.poisson_distribution(self.parameters[0], self.num_k)
        self.model = dl.metrics.meanfields.get(experiment.dynamics_model, degree_dist)


class GNNPEMFMetrics(PoissonEMFMetrics):
    def __init__(self, config, verbose=0):
        super(GNNPEMFMetrics, self).__init__(config, verbose)

    def get_model(self, experiment):
        degree_dist = dl.utilities.poisson_distribution(self.parameters[0], self.num_k)
        self.model = dl.metrics.meanfields.GNN_MF(experiment.model, degree_dist)
