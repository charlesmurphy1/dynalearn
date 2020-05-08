import unittest
from dynalearn.config import ExperimentConfig, MetricsConfig
from dynalearn.experiments import Experiment
from dynalearn.experiments.metrics import TruePESSMetrics, GNNPESSMetrics


class StationaryStateMetricsTest(unittest.TestCase):
    def setUp(self):
        self.experiment = Experiment(ExperimentConfig.test(), verbose=0)

        self.t_metrics = TruePESSMetrics(MetricsConfig.test())
        self.m_metrics = GNNPESSMetrics(MetricsConfig.test())

        self.t_metrics.initialize(self.experiment)
        self.m_metrics.initialize(self.experiment)

    def test_stationary_state(self):

        print(self.t_metrics._stationary_state_())
        print(self.m_metrics._stationary_state_())

    def test_all_stationary_states(self):
        self.t_metrics._all_stationary_states_()
        self.m_metrics._all_stationary_states_()
