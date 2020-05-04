from dynalearn.experiments.summaries import Summary
from abc import abstractmethod


class StationaryStateSummary(Summary):
    @abstractmethod
    def get_metrics(self, experiment):
        raise NotImplemented

    def initialize(self, experiment):
        self.metrics = self.get_metrics(experiment)
        self.data = self.metrics.data


class TruePESSSummary(StationaryStateSummary):
    def get_metrics(self, experiment):
        return experiment.metrics["TruePESSMetrics"]


class GNNPESSSummary(StationaryStateSummary):
    def get_metrics(self, experiment):
        return experiment.metrics["GNNPESSMetrics"]
