from dynalearn.experiments.summaries import Summary


class StatisticsSummary(Summary):
    def initialize(self, experiment):
        self.metrics = experiment.post_metrics["StatisticsMetrics"]
        self.data = self.metrics.data
