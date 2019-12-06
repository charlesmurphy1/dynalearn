from .base_metrics import Metrics
import dynalearn as dl
import numpy as np
import tqdm
from scipy.optimize import bisect


class MeanfieldMetrics(Metrics):
    def __init__(
        self, degree_dist, parameters=None, p_range=None, fp_finder=None, verbose=1
    ):
        self.degree_dist = degree_dist
        self._parameters = parameters
        self.p_range = p_range
        if p_range is None:
            self.p_range = (0.1, 5)

        self.fp_finder = fp_finder
        super(MeanfieldMetrics, self).__init__(verbose)

    def compute(self, experiment):
        mf = dl.meanfields.get(experiment.dynamics_model, self.degree_dist)
        gnn_mf = dl.meanfields.GNN_MF(experiment.model, self.degree_dist)

        true_low_fp = np.zeros((self.parameters.shape[0], mf.s_dim))
        true_high_fp = np.zeros((self.parameters.shape[0], mf.s_dim))
        gnn_low_fp = np.zeros((self.parameters.shape[0], mf.s_dim))
        gnn_high_fp = np.zeros((self.parameters.shape[0], mf.s_dim))

        if self.verbose:
            p_bar = tqdm.tqdm(
                range(2 * len(self.parameters)), "Computing " + self.__class__.__name__
            )
        for i, p in enumerate(self.parameters):
            mf = self.change_param(mf, p)
            fp = self.compute_fixed_points(mf)
            if self.verbose:
                p_bar.update()
            true_low_fp[i] = fp[0]
            true_high_fp[i] = fp[1]

        for i, p in enumerate(self.parameters):
            gnn_mf = self.change_param(gnn_mf, p)
            fp = self.compute_fixed_points(gnn_mf)
            if self.verbose:
                p_bar.update()
            gnn_low_fp[i] = fp[0]
            gnn_high_fp[i] = fp[1]
        if self.verbose:
            p_bar.close()

        self.data[f"true_low_fp"] = true_low_fp
        self.data[f"true_high_fp"] = true_high_fp
        self.data[f"gnn_low_fp"] = gnn_low_fp
        self.data[f"gnn_high_fp"] = gnn_high_fp
        self.data[f"true_thresholds"] = self.compute_thresholds(mf)
        self.data[f"gnn_thresholds"] = self.compute_thresholds(gnn_mf)
        self.data[f"parameters"] = self.parameters

    def compute_fixed_points(self, mf):
        raise NotImplemented()

    def compute_thresholds(self, mf):
        raise NotImplemented()

    def change_param(self, mf, value):
        raise NotImplemented()

    @property
    def parameters(self,):
        if self._parameters is None:
            raise ValueError("No parameter have been given.")
        else:
            return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        self._parameters = parameters


class EpidemicsMFMetrics(MeanfieldMetrics):
    def __init__(
        self,
        degree_dist,
        epsilon=1e-3,
        tol=1e-3,
        parameters=None,
        p_range=None,
        fp_finder=None,
        verbose=1,
    ):
        self.epsilon = epsilon
        self.tol = tol
        super(EpidemicsMFMetrics, self).__init__(
            degree_dist, parameters, p_range, fp_finder, verbose
        )

    def epidemic_state(self, mf):
        x = np.ones(mf.array_shape) * (1 - self.epsilon)
        x[0] = self.epsilon
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

    def compute_thresholds(self, mf):
        def low_f(p):
            _mf = self.change_param(mf, p)
            abs_state = self.absorbing_state(mf).reshape(-1)
            val = _mf.stability(abs_state) - 1
            return val

        def high_f(p):
            _mf = self.change_param(mf, p)
            epi_state = self.epidemic_state(_mf).reshape(-1)
            fp = _mf.search_fixed_point(x0=epi_state, fp_finder=self.fp_finder)
            s = _mf.to_avg(fp)[0]
            if s > 1 - 1e-2:
                val = -1
            else:
                val = 1
            return val

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
    def __init__(
        self,
        num_k=3,
        epsilon=1e-3,
        tol=1e-3,
        parameters=None,
        p_range=None,
        fp_finder=None,
        verbose=1,
    ):
        self.num_k = num_k
        if parameters is None:
            parameters = np.concatenate(
                (np.linspace(0.1, 3, 50), np.linspace(3.1, 10, 20))
            )
        self.degree_dist = dl.meanfields.poisson_distribution(
            parameters[0], num_k=self.num_k
        )
        super(PoissonEpidemicsMFMetrics, self).__init__(
            self.degree_dist, epsilon, tol, parameters, p_range, fp_finder, verbose
        )

    def change_param(self, mf, avgk):
        _degree_dist = dl.meanfields.poisson_distribution(avgk, num_k=self.num_k)
        mf.degree_dist = _degree_dist
        self.degree_dist = _degree_dist
        return mf
