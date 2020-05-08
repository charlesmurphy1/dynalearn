from dynalearn.experiments.summaries import Summary
from abc import abstractmethod


class ForecastSummary(Summary):
    @abstractmethod
    def get_metrics(self, experiment):
        raise NotImplemented

    def initialize(self, experiment):
        self.metrics = self.get_metrics(experiment)
        self.data = self.metrics.data


class TrueForecastSummary(ForecastSummary):
    def get_metrics(self, experiment):
        return experiment.post_metrics["TrueForecastMetrics"]


class GNNForecastSummary(ForecastSummary):
    def get_metrics(self, experiment):
        return experiment.post_metrics["GNNForecastMetrics"]


class TrueRTNForecastSummary(ForecastSummary):
    def get_metrics(self, experiment):
        return experiment.post_metrics["TrueRTNForecastMetrics"]


class GNNRTNForecastSummary(ForecastSummary):
    def get_metrics(self, experiment):
        return experiment.post_metrics["GNNRTNForecastMetrics"]
