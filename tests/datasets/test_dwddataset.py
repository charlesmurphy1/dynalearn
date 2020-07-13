import networkx as nx
import numpy as np
from dynalearn.datasets import (
    DegreeWeightedDiscreteDataset,
    Sampler,
    Data,
    NetworkData,
    WindowedData,
)
from dynalearn.config import DatasetConfig
from unittest import TestCase


class DegreeWeightedDatasetTest(TestCase):
    def setUp(self):
        self.num_networks = 5
        self.num_samples = 6
        self.num_nodes = 10
        self.window_size = 1
        self.window_step = 1
        self.config = DatasetConfig.degree_weighted_discrete_default()
        self.config.window_size = self.window_size
        self.config.window_step = self.window_step
        self.dataset = DegreeWeightedDiscreteDataset(self.config)
        return

    def scenario_1(self):
        networks = NetworkData(
            data=[nx.empty_graph(self.num_nodes) for i in range(self.num_networks)]
        )
        inputs = {
            i: WindowedData(
                data=np.zeros((self.num_samples, self.num_nodes)),
                window_size=self.window_size,
                window_step=self.window_step,
            )
            for i in range(self.num_networks)
        }
        targets = {
            i: Data(data=np.zeros((self.num_samples, self.num_nodes)))
            for i in range(self.num_networks)
        }
        data = {}
        data["networks"] = networks
        data["inputs"] = inputs
        data["targets"] = targets
        self.dataset.data = data

    def scenario_2(self):
        networks = NetworkData(
            data=[nx.complete_graph(self.num_nodes) for i in range(self.num_networks)]
        )
        inputs = {
            i: WindowedData(
                data=np.zeros((self.num_samples, self.num_nodes)),
                window_size=self.window_size,
                window_step=self.window_step,
            )
            for i in range(self.num_networks)
        }
        targets = {
            i: Data(data=np.zeros((self.num_samples, self.num_nodes)))
            for i in range(self.num_networks)
        }
        data = {}
        data["networks"] = networks
        data["inputs"] = inputs
        data["targets"] = targets
        self.dataset.data = data

    def test_get_weights(self):
        self.scenario_1()
        weights = self.dataset._get_weights_()
        x = self.num_networks * self.num_nodes

        for i in range(self.num_networks):
            ref_weights = np.ones((self.num_samples, self.num_nodes)) * x
            np.testing.assert_array_equal(ref_weights, weights[i])

        self.scenario_2()
        weights = self.dataset._get_weights_()
        x = self.num_networks * self.num_nodes

        for i in range(self.num_networks):
            ref_weights = np.ones((self.num_samples, self.num_nodes)) * x
            np.testing.assert_array_equal(ref_weights, weights[i])
