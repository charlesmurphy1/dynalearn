from dynalearn.experiments.summaries import Summary
from abc import abstractmethod


class MeanfieldSummary(Summary):
    @abstractmethod
    def get_metrics(self, experiment):
        raise NotImplemented

    def initialize(self, experiment):
        self.metrics = self.get_metrics(experiment)
        self.data = self.metrics.data


class TruePEMFSummary(MeanfieldSummary):
    def get_metrics(self, experiment):
        return experiment.post_metrics["TruePEMFMetrics"]


class GNNPEMFSummary(MeanfieldSummary):
    def get_metrics(self, experiment):
        return experiment.post_metrics["GNNPEMFMetrics"]
