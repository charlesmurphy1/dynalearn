import unittest
from dynalearn.experiments import Experiment, ExperimentConfig
from dynalearn.experiments.metrics import TruePESSMetrics, MetricsConfig


class StationaryStateMetricsTest(unittest.TestCase):
    def setUp(self):
        self.experiment = Experiment(ExperimentConfig.test(), verbose=0)

        self.metrics = TruePESSMetrics(MetricsConfig.test())

        self.metrics.initialize(self.experiment)

    def test_stationary_state(self):
        self.metrics._stationary_state_()

    def test_all_stationary_states(self):
        self.metrics._all_stationary_states_()
