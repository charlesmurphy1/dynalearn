import networkx as nx
import numpy as np
from dynalearn.experiments.metrics import Metrics
from abc import abstractmethod


class ForecastMetrics(Metrics):
    def __init__(self, config, verbose=0):
        Metrics.__init__(self, config, verbose)
        self.num_forecasts = config.num_forecasts
        self.num_steps = config.num_steps

    @abstractmethod
    def get_model(self, experiment):
        raise NotImplemented

    @abstractmethod
    def initial_state(self):
        raise NotImplemented

    def initialize(self, experiment):
        self.dynamics = experiment.dynamics
        self.networks = experiment.networks
        self.model = self.get_model(experiment)
        self.num_states = self.dynamics.num_states

        self.num_updates = self.num_steps * self.num_forecasts
        self.get_data["forecasts"] = lambda pb: np.array(
            [self._get_forecast_(pb=pb) for i in range(self.num_forecasts)]
        )
        self.names.append("forecasts")

    def _get_forecast_(self, pb=None):
        timeseries = np.zeros((self.num_steps, self.num_states))
        x = self.initial_state()
        self.model.network = self.networks.generate()
        for i in range(self.num_steps):
            timeseries[i] = self.avg(x)
            x = self.model.sample(x)
            if pb is not None:
                pb.update()
        return timeseries

    def avg(self, x):
        avg_x = np.zeros(self.num_states)
        for i in range(self.num_states):
            avg_x[i] = np.mean(x == i)
        return avg_x


class EpidemicsForecastMetrics(ForecastMetrics):
    def __init__(self, config, verbose=0):
        ForecastMetrics.__init__(self, config, verbose=verbose)
        self.initial_infected = config.epsilon

    def initial_state(self):
        return self.dynamics.initial_state(self.initial_infected)


class MetaPopForecastMetrics(ForecastMetrics):
    def __init__(self, config, verbose=0):
        ForecastMetrics.__init__(self, config, verbose=verbose)
        self.initial_infected = config.epsilon

    def initial_state(self):
        state_dist = np.zeros(self.dynamics.num_states)
        state_dist[0] = self.initial_infected
        state_dist[1] = 1 - self.initial_infected
        return self.dynamics.initial_state(state_dist=state_dist)


class RTNForecastMetrics(ForecastMetrics):
    def initialize(self, experiment):
        self.dynamics = experiment.dynamics
        self.networks = experiment.networks
        self.model = self.get_model(experiment)
        self.num_states = self.dynamics.num_states

        self.num_updates = len(self.networks.data) * self.num_steps * self.num_forecasts
        self.get_data["forecasts"] = lambda pb: np.array(
            [self._get_forecast_(pb=pb) for i in range(self.num_forecasts)]
        )
        self.get_data["milestones"] = lambda pb: np.array(
            [self._get_milestone_() for i in range(self.num_forecasts)]
        )
        self.names.append("forecasts")
        self.names.append("milestones")

    def _get_milestone_(self):
        return self.milestones

    def _get_forecast_(self, pb=None):
        x = self.dynamics.initial_state(self.epsilon)
        num_networks = len(self.networks.data)
        timeseries = np.zeros((num_networks * self.num_steps, self.num_states))
        self.networks.time = 0
        self.milestones = []
        k = 0
        for i in range(num_networks):
            self.model.network = self.networks.generate()
            self.milestones.append(k)
            for j in range(self.num_steps):
                timeseries[k] = self.avg(x)
                k += 1
                x = self.model.sample(x)
                if pb is not None:
                    pb.update()
        timeseries = timeseries[timeseries.sum(-1) > 0]
        return timeseries


class TrueForecastMetrics(ForecastMetrics):
    def get_model(self, experiment):
        return experiment.dynamics


class GNNForecastMetrics(ForecastMetrics):
    def get_model(self, experiment):
        return experiment.model


class TrueEpidemicsForecastMetrics(TrueForecastMetrics, EpidemicsForecastMetrics):
    def __init__(self, config, verbose=0):
        TrueForecastMetrics.__init__(self, config, verbose=verbose)
        EpidemicsForecastMetrics.__init__(self, config, verbose=verbose)

    def get_model(self, experiment):
        return experiment.dynamics


class GNNEpidemicsForecastMetrics(GNNForecastMetrics, EpidemicsForecastMetrics):
    def __init__(self, config, verbose=0):
        GNNForecastMetrics.__init__(self, config, verbose=verbose)
        EpidemicsForecastMetrics.__init__(self, config, verbose=verbose)

    def get_model(self, experiment):
        return experiment.model


class TrueMetaPopForecastMetrics(TrueForecastMetrics, MetaPopForecastMetrics):
    def __init__(self, config, verbose=0):
        TrueForecastMetrics.__init__(self, config, verbose=verbose)
        MetaPopForecastMetrics.__init__(self, config, verbose=verbose)

    def get_model(self, experiment):
        return experiment.dynamics


class GNNMetaPopForecastMetrics(GNNForecastMetrics, MetaPopForecastMetrics):
    def __init__(self, config, verbose=0):
        GNNForecastMetrics.__init__(self, config, verbose=verbose)
        MetaPopForecastMetrics.__init__(self, config, verbose=verbose)

    def get_model(self, experiment):
        return experiment.model


class TrueRTNForecastMetrics(RTNForecastMetrics, TrueForecastMetrics):
    def __init__(self, config, verbose=0):
        RTNForecastMetrics.__init__(self, config, verbose)
        TrueForecastMetrics.__init__(self, config, verbose)


class GNNRTNForecastMetrics(RTNForecastMetrics, GNNForecastMetrics):
    def __init__(self, config, verbose=0):
        RTNForecastMetrics.__init__(self, config, verbose)
        GNNForecastMetrics.__init__(self, config, verbose)
