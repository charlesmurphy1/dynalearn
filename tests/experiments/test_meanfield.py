import unittest
from dynalearn.experiments import Experiment, ExperimentConfig
from dynalearn.experiments.metrics import TruePEMFMetrics, MetricsConfig


class MeanfieldMetricsTest(unittest.TestCase):
    def setUp(self):
        self.experiment = Experiment(ExperimentConfig.test(), verbose=0)

        self.metrics = TruePEMFMetrics(MetricsConfig.test())

        self.metrics.initialize(self.experiment)

    def test_fixed_point(self):
        self.metrics._fixed_point_()

    def test_all_fixed_points(self):
        self.metrics._all_fixed_points_()
