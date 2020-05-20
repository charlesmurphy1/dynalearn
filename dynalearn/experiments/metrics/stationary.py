import networkx as nx
import numpy as np

from abc import abstractmethod
from .metrics import Metrics
from dynalearn.utilities import poisson_distribution
from dynalearn.networks import ConfigurationNetwork, RealTemporalNetwork
from dynalearn.config import NetworkConfig
from random import sample


class StationaryStateMetrics(Metrics):
    def __init__(self, config, verbose=0):
        Metrics.__init__(self, config, verbose)

        if "parameters" in config.__dict__:
            self.parameters = config.parameters
        else:
            self.parameters = None
        self.num_samples = config.num_samples
        self.full_data = config.full_data
        self.burn = config.burn

        self.dynamics = None
        self.networks = None
        self.model = None

        if self.parameters is None:
            self.names = ["stationary_state"]
        else:
            self.names = ["parameters", "stationary_state"]

        if self.full_data:
            self.statistics = self._statistics_full
        else:
            self.statistics = self._statistics_meanstd

    @abstractmethod
    def get_model(self, experiment):
        raise NotImplementedError()

    def get_networks(self, experiment):
        return experiment.networks

    def change_param(self, p):
        return

    def initialize(self, experiment):
        self.dynamics = experiment.dynamics
        self.networks = self.get_networks(experiment)
        self.model = self.get_model(experiment)

        if self.parameters is None:
            self.num_updates = self.num_samples
            self.get_data["stationary_state"] = lambda pb: self._stationary_state_(
                pb=pb
            )
        else:
            self.num_updates = self.num_samples * len(self.parameters)
            self.get_data["parameters"] = lambda pb=None: self.parameters
            self.get_data["stationary_state"] = lambda pb: self._all_stationary_states_(
                pb
            )

    def _all_stationary_states_(self, pb=None):
        stationary_states = []
        for p in self.parameters:
            stationary_states.append(self._stationary_state_(param=p, pb=pb))
        return np.array(stationary_states)

    def _stationary_state_(self, param=None, epsilon=None, pb=None):
        if param is not None:
            self.change_param(param)
        samples = np.zeros((self.num_samples, self.networks.num_nodes))
        for i in range(self.num_samples):
            if issubclass(self.networks.__class__, RealTemporalNetwork):
                g = self.networks.complete_network
            elif hasattr(self.networks, "generate"):
                g = self.networks.generate()
            else:
                g = sample(self.networks.data, 1)[0]
            self.dynamics.network = g
            self.model.network = g
            x0 = self.dynamics.initial_state(initial_infected=epsilon)
            samples[i] = self.burning(x0, self.burn)
            if self.verbose and pb is not None:
                pb.update()
        avg_samples = self.avg(samples, axis=-1)
        return self.statistics(avg_samples)

    def burning(self, x, burn=1):
        for b in range(burn):
            x = self.model.sample(x)
        return x

    def avg(self, x, axis=None):
        avg_x = []
        for i in range(self.model.num_states):
            avg_x.append(np.mean(x == i, axis=axis))
        return np.array(avg_x)

    def _statistics_meanstd(self, avg_samples):
        data = np.zeros((2, avg_samples.shape[0]))
        data[0] = np.mean(avg_samples, axis=-1)
        data[1] = np.std(avg_samples, axis=-1)
        return data

    def _statistics_full(self, avg_samples):
        data = avg_samples.T
        return data


class TrueSSMetrics(StationaryStateMetrics):
    def get_model(self, experiment):
        return experiment.dynamics


class GNNSSMetrics(StationaryStateMetrics):
    def get_model(self, experiment):
        return experiment.model


class EpidemicSSMetrics(StationaryStateMetrics):
    def __init__(self, config, verbose=0):
        StationaryStateMetrics.__init__(self, config, verbose)
        self.epsilon = config.epsilon

        if self.parameters is None:
            self.names = ["absorbing_stationary_state", "epidemic_stationary_state"]
        else:
            self.names = [
                "parameters",
                "absorbing_stationary_state",
                "epidemic_stationary_state",
            ]

    def initialize(self, experiment):
        self.dynamics = experiment.dynamics
        self.networks = self.get_networks(experiment)
        self.model = self.get_model(experiment)

        if self.parameters is None:
            self.num_updates = 2 * self.num_samples
            self.get_data[
                "absorbing_stationary_state"
            ] = lambda pb: self._stationary_state_(self.epsilon, pb=pb)
            self.get_data[
                "epidemic_stationary_state"
            ] = lambda pb: self._stationary_state_(1 - self.epsilon, pb=pb)
        else:
            self.num_updates = 2 * self.num_samples * len(self.parameters)
            self.get_data["parameters"] = lambda pb=None: self.parameters
            self.get_data[
                "absorbing_stationary_state"
            ] = lambda pb: self._all_stationary_states_(
                epsilon=self.epsilon, pb=pb, ascend=True
            )
            self.get_data[
                "epidemic_stationary_state"
            ] = lambda pb: self._all_stationary_states_(
                epsilon=1 - self.epsilon, pb=pb, ascend=False
            )

    def _all_stationary_states_(self, epsilon=None, pb=None, ascend=True):
        stationary_states = []
        if ascend:
            params = np.sort(self.parameters)
        else:
            params = np.sort(self.parameters)[::-1]

        for p in params:
            stats = self._stationary_state_(p, epsilon, pb)
            stationary_states.append(stats)
            if self.full_data:
                epsilon = 1 - np.mean(stats, axis=-1)[0]
            else:
                epsilon = 1 - stats[0, 0]

            if epsilon < self.epsilon:
                epsilon = self.epsilon
            elif epsilon > 1 - self.epsilon:
                epsilon = 1 - self.epsilon
        if ascend:
            return np.array(stationary_states)
        else:
            return np.array(stationary_states)[::-1]


class TrueESSMetrics(TrueSSMetrics, EpidemicSSMetrics):
    def __init__(self, config, verbose=0):
        TrueSSMetrics.__init__(self, config, verbose)
        EpidemicSSMetrics.__init__(self, config, verbose)


class GNNESSMetrics(GNNSSMetrics, EpidemicSSMetrics):
    def __init__(self, config, verbose=0):
        GNNSSMetrics.__init__(self, config, verbose)
        EpidemicSSMetrics.__init__(self, config, verbose)


class PoissonESSMetrics(EpidemicSSMetrics):
    def __init__(self, config, verbose=0):
        EpidemicSSMetrics.__init__(self, config, verbose)
        self.num_nodes = config.num_nodes
        self.num_k = config.num_k

    def get_networks(self, experiment):
        p_k = poisson_distribution(self.parameters[0], self.num_k)
        config = NetworkConfig.configuration(self.num_nodes, p_k)
        return ConfigurationNetwork(config)

    def change_param(self, avgk):
        p_k = poisson_distribution(avgk, self.num_k)
        config = NetworkConfig.configuration(self.num_nodes, p_k)
        self.networks = ConfigurationNetwork(config)


class TruePESSMetrics(TrueSSMetrics, PoissonESSMetrics):
    def __init__(self, config, verbose=0):
        TrueSSMetrics.__init__(self, config, verbose)
        PoissonESSMetrics.__init__(self, config, verbose)


class GNNPESSMetrics(GNNSSMetrics, PoissonESSMetrics):
    def __init__(self, config, verbose=0):
        GNNSSMetrics.__init__(self, config, verbose)
        PoissonESSMetrics.__init__(self, config, verbose)
