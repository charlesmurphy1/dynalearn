from .base import Metrics
import dynalearn as dl
import numpy as np
import tqdm
from abc import abstractmethod


def bisection(f, a, b, tol=1e-3, p_bar=None, maxiter=100):

    if f(a) * f(b) >= 0:
        print("Warning: f(a) and f(b) have the same sign")
        return (a + b) / 2

    for i in range(maxiter):
        x = (a + b) / 2
        if f(x) > 0:
            b = x
        else:
            a = x
        diff = np.abs(a - b)
        if p_bar is not None:
            p_bar.set_description("diff: {0}".format(diff))
            p_bar.update()
        if diff < tol:
            return x

    print("Warning: bisection did not converge.")
    return x


class MeanfieldMetrics(Metrics):
    def __init__(self, degree_dist, config, verbose=1):
        self.degree_dist = degree_dist
        self.__config = config
        self.parameters = config.mf_parameters
        self.p_range = config.p_range
        self.fp_finder = config.fp_finder
        super(MeanfieldMetrics, self).__init__(verbose)

    def compute(self, experiment):
        true_mf = dl.metrics.meanfields.get(experiment.dynamics_model, self.degree_dist)
        gnn_mf = dl.metrics.meanfields.GNN_MF(experiment.model, self.degree_dist)

        if self.verbose:
            print("Computing " + self.__class__.__name__ + ": True")
        self.data[f"true_thresholds"] = self.compute_thresholds(true_mf)
        self.data[f"true_fp"] = self.compute_fixed_points(true_mf)
        if self.verbose:
            print("Computing " + self.__class__.__name__ + ": GNN")
        self.data[f"gnn_thresholds"] = self.compute_thresholds(gnn_mf)
        self.data[f"gnn_fp"] = self.compute_fixed_points(gnn_mf)
        self.data[f"parameters"] = self.parameters

    @abstractmethod
    def compute_fixed_points(self, mf):
        raise NotImplementedError("compute_fixed_points must be implemented.")

    @abstractmethod
    def compute_thresholds(self, mf):
        raise NotImplementedError("compute_thresholds must be implemented.")

    @abstractmethod
    def change_param(self, mf, value):
        raise NotImplementedError("change_param must be implemented.")


class EpidemicsMFMetrics(MeanfieldMetrics):
    def __init__(self, degree_dist, config, verbose=1):
        self.epsilon = config.epsilon
        self.tol = config.tol
        if config.discontinuous:
            self.criterion = self.__discontinuous_threshold_criterion
        else:
            self.criterion = self.__continuous_threshold_criterion

        super(EpidemicsMFMetrics, self).__init__(degree_dist, config, verbose)

    def compute_fixed_points(self, mf):

        size = (len(self.parameters), mf.s_dim)
        x0 = self.absorbing_state(mf)
        low_fp = np.zeros(size)
        if self.verbose:
            p_bar = tqdm.tqdm(range(2 * len(self.parameters)), "Fixed points")

        for i, p in enumerate(self.parameters):
            mf = self.change_param(mf, p)
            x0 = mf.search_fixed_point(x0=x0, fp_finder=self.fp_finder)
            low_fp[i] = mf.to_avg(x0)
            if self.verbose:
                p_bar.update()

        high_fp = np.zeros(size)
        x0 = self.epidemic_state(mf)
        for i, p in reversed(list(enumerate(self.parameters))):
            mf = self.change_param(mf, p)
            x0 = mf.search_fixed_point(x0=x0, fp_finder=self.fp_finder)
            high_fp[i] = mf.to_avg(x0)
            if self.verbose:
                p_bar.update()
        if self.verbose:
            p_bar.close()

        return np.concatenate(
            (low_fp.reshape(1, *size), high_fp.reshape(1, *size)), axis=0
        )

    def compute_thresholds(self, mf):
        low_f = lambda p: self.criterion(mf, self.absorbing_state(mf).reshape(-1), p)
        high_f = lambda p: self.criterion(mf, self.epidemic_state(mf).reshape(-1), p)

        thresholds = np.array([])
        p_min = self.p_range[0]
        p_max = self.p_range[1]

        if self.verbose:
            p_bar = tqdm.tqdm(range(1), "Thresholds")

        if low_f(p_min) * low_f(p_max) < 0:
            low_threshold = bisection(low_f, p_min, p_max, tol=self.tol, p_bar=p_bar)
            thresholds = np.append(thresholds, low_threshold)

        if high_f(p_min) * high_f(p_max) < 0:
            high_threshold = bisection(high_f, p_min, p_max, tol=self.tol, p_bar=p_bar)
            thresholds = np.append(thresholds, high_threshold)

        if self.verbose:
            p_bar.close()
        return thresholds

    def epidemic_state(self, mf):
        x = np.ones(mf.array_shape).astype(mf.dtype) * self.epsilon
        x[0] = 1 - self.epsilon
        x = 1 - x
        return mf.normalize_state(x)

    def absorbing_state(self, mf):
        x = np.ones(mf.array_shape) * self.epsilon
        x[0] = 1 - self.epsilon
        return mf.normalize_state(x)

    def __continuous_threshold_criterion(self, mf, states, p):
        _mf = self.change_param(mf, p)
        val = _mf.stability(states) - 1
        return val

    def __discontinuous_threshold_criterion(self, mf, states, p):
        _mf = self.change_param(mf, p)
        fp = _mf.search_fixed_point(x0=states, fp_finder=self.fp_finder)
        s = _mf.to_avg(fp)[0]
        if s > 1 - 1e-2:
            val = -1
        else:
            val = 1
        return val


class PoissonEpidemicsMFMetrics(EpidemicsMFMetrics):
    def __init__(self, config, verbose=1):
        self.num_k = config.num_k
        self.degree_dist = dl.utilities.poisson_distribution(
            config.mf_parameters[0], num_k=self.num_k
        )
        super(PoissonEpidemicsMFMetrics, self).__init__(
            self.degree_dist, config, verbose
        )

    def change_param(self, mf, avgk):
        _degree_dist = dl.utilities.poisson_distribution(avgk, self.num_k)
        mf.degree_dist = _degree_dist
        self.degree_dist = _degree_dist
        return mf


class DegreeRegularEpidemicsMFMetrics(EpidemicsMFMetrics):
    def __init__(self, config, verbose=1):
        self.degree_dist = dl.utilities.kronecker_distribution(config.mf_parameters[0])
        super(DegreeRegularEpidemicsMFMetrics, self).__init__(
            self.degree_dist, config, verbose
        )

    def change_param(self, mf, avgk):
        _degree_dist = dl.utilities.kronecker_distribution(avgk)
        mf.degree_dist = _degree_dist
        self.degree_dist = _degree_dist
        return mf
