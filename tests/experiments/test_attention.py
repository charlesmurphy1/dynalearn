import unittest
import numpy as np
from dynalearn.config import ExperimentConfig
from dynalearn.experiments import Experiment


class AttentionMetricsTest(unittest.TestCase):
    @property
    def name(self):
        return "AttentionMetrics"

    def setUp(self):
        self.config = ExperimentConfig.metapop("test", "dsir", "w_ba")
        self.config.networks.num_nodes = 10
        self.config.train_details.num_samples = 10
        self.config.train_details.num_networks = 1
        self.config.metrics.names = [self.name]
        self.config.metrics.num_steps = [1, 7, 14]
        self.config.model.heads = 1
        self.exp = Experiment(self.config)
        self.exp.generate_data()
        self.exp.partition_test_dataset()
        self.exp.partition_val_dataset()

    def test_compute(self):
        self.exp.metrics[self.name].compute(self.exp)


class AttentionNMIMetricsTest(unittest.TestCase):
    @property
    def name(self):
        return "AttentionNMIMetrics"

    def setUp(self):
        self.config = ExperimentConfig.metapop("test", "dsir", "w_ba")
        self.config.networks.num_nodes = 10
        self.config.train_details.num_samples = 10
        self.config.train_details.num_networks = 1
        self.config.metrics.names = [self.name]
        self.config.metrics.num_steps = [1, 7, 14]
        self.config.model.heads = 1
        self.exp = Experiment(self.config)
        self.exp.generate_data()
        self.exp.partition_test_dataset()
        self.exp.partition_val_dataset()

    def test_compute(self):
        self.exp.metrics[self.name].compute(self.exp)
        # for k, v in self.exp.metrics[self.name].data.items():
        #     print(k, v)
