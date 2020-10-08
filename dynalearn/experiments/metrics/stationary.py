import numpy as np

from abc import abstractmethod
from random import sample
from dynalearn.utilities import poisson_distribution
from dynalearn.networks import ConfigurationNetwork, ERNetwork
from dynalearn.config import NetworkConfig
from .metrics import Metrics
from ._utils import Initializer, ModelSampler, Statistics


class StationaryStateMetrics(Metrics):
    def __init__(self, config, verbose=0):
        Metrics.__init__(self, config, verbose)
        if "parameters" in config.__dict__:
            self.parameters = config.parameters
        else:
            self.parameters = None
        self.num_samples = config.num_samples
        self.initializer = Initializer(self.config)
        self.sampler = ModelSampler.getter(self.config)
        self.statistics = Statistics.getter(self.config)

        self.dynamics = None
        self.networks = None
        self.model = None

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
        self.initializer.setUp(self)
        self.sampler.setUp(self)

        self.num_updates = self.num_samples

        if self.parameters is not None:
            self.num_updates *= len(self.parameters)
            self.names.append("parameters")
            self.get_data["parameters"] = lambda pb: self.parameters
        for m in self.initializer.all_modes:
            self.get_data[m] = lambda pb: self._all_stationary_states_(mode=m, pb=pb)
            self.names.append(m)

    def initialize_network(self):
        g = self.networks.generate()
        self.dynamics.network = g
        self.model.network = g

    def _stationary_(self, param=None, pb=None):
        if param is not None:
            self.change_param(param)
        samples = []
        for i in range(self.num_samples):
            self.initialize_network()
            samples.append(self.sampler(self.model, self.initializer, self.statistics))
            if pb is not None:
                pb.update()
        samples = np.array(samples)
        samples.reshape(-1, samples.shape[-1])
        self.initializer.update(self.statistics.avg(samples))
        y = self.statistics(samples)
        return y

    def _all_stationary_states_(self, mode=None, pb=None):
        s = []
        if mode is not None:
            self.initializer.mode = mode
        if self.parameters is not None:
            for p in self.parameters:
                s.append(self._stationary_(param=p, pb=pb))
        else:
            s.append(self._stationary_(pb=pb))
        return np.array(s)


class TrueSSMetrics(StationaryStateMetrics):
    def get_model(self, experiment):
        return experiment.dynamics


class GNNSSMetrics(StationaryStateMetrics):
    def get_model(self, experiment):
        return experiment.model


class PoissonSSMetrics(StationaryStateMetrics):
    def __init__(self, config, verbose=0):
        StationaryStateMetrics.__init__(self, config, verbose)
        self.num_nodes = config.num_nodes
        self.num_k = config.num_k

    def get_networks(self, experiment):
        p_k = poisson_distribution(self.parameters[0], self.num_k)
        config = NetworkConfig.configuration(self.num_nodes, p_k)
        self.weight_gen = experiment.networks.weight_gen
        return ConfigurationNetwork(config, weight_gen=self.weight_gen)

    def change_param(self, avgk):
        p_k = poisson_distribution(avgk, self.num_k)
        config = NetworkConfig.configuration(self.num_nodes, p_k)
        self.networks = ConfigurationNetwork(config, weight_gen=self.weight_gen)


class TruePSSMetrics(TrueSSMetrics, PoissonSSMetrics):
    def __init__(self, config, verbose=0):
        TrueSSMetrics.__init__(self, config, verbose)
        PoissonSSMetrics.__init__(self, config, verbose)


class GNNPSSMetrics(GNNSSMetrics, PoissonSSMetrics):
    def __init__(self, config, verbose=0):
        GNNSSMetrics.__init__(self, config, verbose)
        PoissonSSMetrics.__init__(self, config, verbose)


class ErdosRenyiSSMetrics(StationaryStateMetrics):
    def __init__(self, config, verbose=0):
        StationaryStateMetrics.__init__(self, config, verbose)
        self.num_nodes = config.num_nodes

    def get_networks(self, experiment):
        p = self.parameters[0] / (self.num_nodes - 1)
        config = NetworkConfig.erdosrenyi(self.num_nodes, p)
        self.weight_gen = experiment.networks.weight_gen
        return ERNetwork(config, weight_gen=self.weight_gen)

    def change_param(self, avgk):
        self.networks.config.p = avgk / (self.num_nodes - 1)


class TrueERSSMetrics(TrueSSMetrics, ErdosRenyiSSMetrics):
    def __init__(self, config, verbose=0):
        TrueSSMetrics.__init__(self, config, verbose)
        ErdosRenyiSSMetrics.__init__(self, config, verbose)


class GNNERSSMetrics(GNNSSMetrics, ErdosRenyiSSMetrics):
    def __init__(self, config, verbose=0):
        GNNSSMetrics.__init__(self, config, verbose)
        ErdosRenyiSSMetrics.__init__(self, config, verbose)
