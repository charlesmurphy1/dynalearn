import networkx as nx
import numpy as np

from abc import abstractmethod
from .metrics import Metrics
from dynalearn.meanfields.getter import get as get_mf
from dynalearn.meanfields.finder import get as get_finder
from dynalearn.utilities import poisson_distribution
from random import sample


class MeanfieldMetrics(Metrics):
    def __init__(self, config, verbose=0):
        Metrics.__init__(self, config, verbose)

        if "parameters" in config.__dict__:
            self.parameters = config.parameters
        else:
            self.parameters = None

        self.p_k = None
        self.model = None
        self.mf = None
        self.finder = get_finder(config)

        if self.parameters is None:
            self.names = ["fixed_points"]
        else:
            self.names = ["parameters", "fixed_points"]

    @abstractmethod
    def get_model(self, experiment):
        raise NotImplementedError()

    def change_param(self, p):
        return

    def initial_state(self, epsilon=None):
        return self.mf.random_state()

    def initialize(self, experiment):
        self.model = self.get_model(experiment)
        self.p_k = self.get_degreedist(experiment)
        self.mf = get_mf(self.model)(self.p_k)
        self.mf.with_numba = self.config.with_numba
        if self.parameters is None:
            self.num_updates = 1
            self.get_data["fixed_point"] = lambda pb: self._fixed_point_(pb=pb)
        else:
            self.num_updates = len(self.parameters)
            self.get_data["parameters"] = lambda pd=None: self.parameters
            self.get_data["fixed_point"] = lambda pb: self._all_fixed_points_(pb=pb)

    def _fixed_point_(self, param=None, epsilon=None):
        if param is not None:
            self.change_param(param)

        x0 = self.mf.flatten(self.initial_state(epsilon))

        f = lambda x: self.mf.flatten(self.mf.update(self.mf.unflatten(x)))

        result = self.finder(f, x0)
        if self.verbose != 0 and not result.success:
            if param is not None:
                print(f"Fixed point has not converged with parameter {param}.")
            else:
                print(f"Fixed point has not converged.")
        x = self.mf.unflatten(result.x)

        return self.mf.avg(x)

    def _all_fixed_points_(self, pb=None):
        fixed_points = []
        for p in self.parameters:
            fixed_points.append(self._fixed_point_(param=p))
            if pb is not None:
                pb.update()
        return np.array(fixed_points)


class TrueMFMetrics(MeanfieldMetrics):
    def get_model(self, experiment):
        return experiment.dynamics


class GNNMFMetrics(MeanfieldMetrics):
    def get_model(self, experiment):
        return experiment.model


class EpidemicMFMetrics(MeanfieldMetrics):
    def __init__(self, config, verbose=0):
        MeanfieldMetrics.__init__(self, config, verbose)
        self.epsilon = config.epsilon

        if self.parameters is None:
            self.names = ["absorbing_fixed_point", "epidemic_fixed_point"]
        else:
            self.names = ["parameters", "absorbing_fixed_point", "epidemic_fixed_point"]

    def initial_state(self, epsilon=None):
        x = self.mf.random_state()
        if epsilon is None:
            return x
        for k in x:
            x[k] = np.ones(self.model.num_states) * epsilon / self.model.num_states
            x[k][0] = 1 - epsilon
        return self.mf.normalize_state(x)

    def initialize(self, experiment):
        self.model = self.get_model(experiment)
        self.mf = get_mf(self.model)(self.p_k)

        if self.parameters is None:
            self.num_updates = 2
            self.get_data["absorbing_fixed_point"] = lambda pb: self._fixed_point_(
                epsilon=self.epsilon, pb=pb
            )
            self.get_data["epidemic_fixed_point"] = lambda pb: self._fixed_point_(
                epsilon=1 - self.epsilon, pb=pb
            )
        else:
            self.num_updates = 2 * len(self.parameters)
            self.get_data["parameters"] = lambda pb=None: self.parameters
            self.get_data["absorbing_fixed_point"] = lambda pb: self._all_fixed_points_(
                epsilon=self.epsilon, pb=pb
            )
            self.get_data["epidemic_fixed_point"] = lambda pb: self._all_fixed_points_(
                epsilon=1 - self.epsilon, pb=pb
            )

    def _all_fixed_points_(self, epsilon=None, pb=None):
        fixed_points = []
        for p in self.parameters:
            fp = self._fixed_point_(param=p)
            fixed_points.append(fp)
            epsilon = 1 - fp[0]
            if pb is not None:
                pb.update()
        return np.array(fixed_points)


class PoissonEMFMetrics(EpidemicMFMetrics):
    def __init__(self, config, verbose=0):
        EpidemicMFMetrics.__init__(self, config, verbose)
        self.num_k = config.num_k
        self.p_k = poisson_distribution(1, self.num_k)

    def change_param(self, avgk):
        self.p_k = poisson_distribution(avgk, self.num_k)
        self.mf.p_k = self.p_k


class TruePEMFMetrics(TrueMFMetrics, PoissonEMFMetrics):
    def __init__(self, config, verbose=0):
        TrueMFMetrics.__init__(self, config, verbose)
        PoissonEMFMetrics.__init__(self, config, verbose)


class GNNPEMFMetrics(GNNMFMetrics, PoissonEMFMetrics):
    def __init__(self, config, verbose=0):
        GNNMFMetrics.__init__(self, config, verbose)
        PoissonEMFMetrics.__init__(self, config, verbose)
