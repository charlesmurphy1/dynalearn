from .base import Metrics
import dynalearn as dl
import numpy as np
import tqdm
from scipy.optimize import bisect
from abc import abstractmethod


class MeanfieldMetrics(Metrics):
    def __init__(self, degree_dist, config, verbose=1):
        self.degree_dist = degree_dist
        self.__config = config
        self.parameters = config.mf_parameters
        self.p_range = config.p_range
        self.fp_finder = config.fp_finder
        super(MeanfieldMetrics, self).__init__(verbose)

    def compute(self, experiment):
        mf = dl.metrics.meanfields.get(experiment.dynamics_model, self.degree_dist)
        gnn_mf = dl.metrics.meanfields.GNN_MF(experiment.model, self.degree_dist)

        true_low_fp = np.zeros((self.parameters.shape[0], mf.s_dim))
        true_high_fp = np.zeros((self.parameters.shape[0], mf.s_dim))
        gnn_low_fp = np.zeros((self.parameters.shape[0], mf.s_dim))
        gnn_high_fp = np.zeros((self.parameters.shape[0], mf.s_dim))

        if self.verbose:
            print("Computing " + self.__class__.__name__)
            p_bar = tqdm.tqdm(range(len(self.parameters)), "Fixed points: True model")
        for i, p in enumerate(self.parameters):
            mf = self.change_param(mf, p)
            fp = self.compute_fixed_points(mf)
            if self.verbose:
                p_bar.update()
            true_low_fp[i] = fp[0]
            true_high_fp[i] = fp[1]

        if self.verbose:
            p_bar.close()
            p_bar = tqdm.tqdm(range(len(self.parameters)), "Fixed points: GNN model")
        for i, p in enumerate(self.parameters):
            gnn_mf = self.change_param(gnn_mf, p)
            fp = self.compute_fixed_points(gnn_mf)
            if self.verbose:
                p_bar.update()
            gnn_low_fp[i] = fp[0]
            gnn_high_fp[i] = fp[1]

        self.data[f"true_low_fp"] = true_low_fp
        self.data[f"true_high_fp"] = true_high_fp
        self.data[f"gnn_low_fp"] = gnn_low_fp
        self.data[f"gnn_high_fp"] = gnn_high_fp

        if self.verbose:
            p_bar.close()
            print("Thresholds: True model")
        self.data[f"true_thresholds"] = self.compute_thresholds(mf)

        if self.verbose:
            print("Thresholds: GNN model")
        self.data[f"gnn_thresholds"] = self.compute_thresholds(gnn_mf)
        self.data[f"parameters"] = self.parameters

        if self.verbose:
            p_bar.close()

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

    def epidemic_state(self, mf):
        x = np.ones(mf.array_shape).astype(mf.dtype) * self.epsilon
        x[0] = 1 - self.epsilon
        x = 1 - x
        return mf.normalize_state(x)

    def absorbing_state(self, mf):
        x = np.ones(mf.array_shape) * self.epsilon
        x[0] = 1 - self.epsilon
        return mf.normalize_state(x)

    def compute_fixed_points(self, mf):

        abs_state = self.absorbing_state(mf)
        low_fp = mf.to_avg(
            mf.search_fixed_point(x0=abs_state, fp_finder=self.fp_finder)
        )

        epi_state = self.epidemic_state(mf)
        high_fp = mf.to_avg(
            mf.search_fixed_point(x0=epi_state, fp_finder=self.fp_finder)
        )

        fp = np.array([low_fp, high_fp])

        return fp

    def __continuous_threshold_criterion(self, mf, states, p):
        _mf = self.change_param(mf, p)
        val = _mf.stability(abs_state) - 1
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

    def compute_thresholds(self, mf):
        low_f = lambda p: self.criterion(mf, self.absorbing_state(mf).reshape(-1), p)
        high_f = lambda p: self.criterion(mf, self.epidemic_state(mf).reshape(-1), p)

        thresholds = np.array([])
        p_min = self.p_range[0]
        p_max = self.p_range[1]

        if low_f(p_min) * low_f(p_max) < 0:
            low_threshold = bisect(low_f, p_min, p_max, xtol=self.tol)
            thresholds = np.append(thresholds, low_threshold)
        else:
            print(
                "invalid values for low thresholds: {0}".format(
                    [low_f(p_min), low_f(p_max)]
                )
            )

        if high_f(p_min) * high_f(p_max) < 0:
            high_threshold = bisect(high_f, p_min, p_max, xtol=self.tol)
            thresholds = np.append(thresholds, high_threshold)
        else:
            print(
                "invalid values for high thresholds: {0}".format(
                    [high_f(p_min), high_f(p_max)]
                )
            )

        return thresholds


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
