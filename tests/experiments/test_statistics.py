import networkx as nx
import numpy as np
from unittest import TestCase
from dynalearn.dynamics import SIS
from dynalearn.experiments.metrics import StatisticsMetrics, MetricsConfig
from dynalearn.config import DynamicsConfig


class TestStatisticsMetrics(TestCase):
    def setUp(self):
        self.n = 100
        self.p = 0.05
        self.num_samples = 5
        self.num_networks = 2

        d_config = DynamicsConfig.sis_default()
        m_config = MetricsConfig.sis_fast()

        self.dynamics = SIS(d_config)
        self.metrics = StatisticsMetrics(m_config)
        self.networks = {}
        self.networks[0] = nx.empty_graph(self.n)
        self.networks[1] = nx.complete_graph(self.n)
        self.all_nodes = {
            i: {t: list(g.nodes()) for t in range(self.num_samples)}
            for i, g in self.networks.items()
        }
        self.num_points = {i: self.num_samples for i in self.networks}

        self.metrics.networks = self.networks
        self.metrics.all_nodes = self.all_nodes
        self.metrics.num_points = self.num_points
        self.metrics.num_states = 2

        self.inputs_zeros = {
            i: np.zeros((self.num_samples, self.n)) for i in range(self.num_networks)
        }
        self.targets_zeros = {
            i: np.zeros((self.num_samples, self.n)) for i in range(self.num_networks)
        }
        self.inputs_ones = {
            i: np.ones((self.num_samples, self.n)) for i in range(self.num_networks)
        }
        self.targets_ones = {
            i: np.ones((self.num_samples, self.n)) for i in range(self.num_networks)
        }
        self.inputs_rand = {
            i: np.random.randint(2, size=(self.num_samples, self.n))
            for i in range(self.num_networks)
        }
        self.targets_rand = {
            i: np.random.randint(2, size=(self.num_samples, self.n))
            for i in range(self.num_networks)
        }

    def test_getsumm_zeros(self):
        self.metrics.inputs = self.inputs_zeros
        self.metrics.targets = self.targets_zeros

        summaries = self.metrics._get_summaries_()
        ref_summaries = np.array([[0, 0, 0], [0, self.n - 1, 0]])
        is_equal = True
        for s in summaries:
            if np.all(s != ref_summaries):
                is_equal = False
                break
        self.assertTrue(is_equal)

    def test_getsumm_ones(self):
        self.metrics.inputs = self.inputs_ones
        self.metrics.targets = self.targets_ones

        summaries = self.metrics._get_summaries_()
        ref_summaries = np.array([[1, 0, 0], [1, 0, self.n - 1]])
        is_equal = True
        for s in summaries:
            if np.all(s != ref_summaries):
                is_equal = False
                break
        self.assertTrue(is_equal)

    def test_getsumm_rand(self):
        self.metrics.inputs = self.inputs_rand
        self.metrics.targets = self.targets_rand

        summaries = self.metrics._get_summaries_()
        num_inf = self.inputs_rand[1].sum().astype("int")
        ref_summaries = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, self.n - num_inf - 1, num_inf],
                [1, self.n - num_inf, num_inf - 1],
            ]
        )
        is_equal = True
        for s in summaries:
            if np.all(s != ref_summaries):
                is_equal = False
                break
        self.assertTrue(is_equal)

    def test_getstats_zeros(self):
        self.metrics.inputs = self.inputs_zeros
        self.metrics.targets = self.targets_zeros

        summaries = self.metrics._get_summaries_()
        ref_stats = np.ones(summaries.shape[0]) * self.n * self.num_samples
        stats = self.metrics._get_stats_(self.all_nodes)
        self.assertTrue(np.all(ref_stats == stats))

    def test_getstats_ones(self):
        self.metrics.inputs = self.inputs_ones
        self.metrics.targets = self.targets_ones

        summaries = self.metrics._get_summaries_()
        ref_stats = np.ones(summaries.shape[0]) * self.n * self.num_samples
        stats = self.metrics._get_stats_(self.all_nodes)
        self.assertTrue(np.all(ref_stats == stats))
