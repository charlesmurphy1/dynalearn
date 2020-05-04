from dynalearn.experiments.summaries import Summary


class StatisticsSummary(Summary):
    def initialize(self, experiment):
        self.metrics = experiment.metrics["StatisticsMetrics"]
        self.data = self.metrics.data
