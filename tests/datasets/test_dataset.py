import networkx as nx
import numpy as np
import torch

from dynalearn.datasets import Dataset, Sampler
from dynalearn.config import DatasetConfig
from unittest import TestCase


class DatasetTest(TestCase):
    def setUp(self):
        self.config = DatasetConfig.plain_default()
        self.dataset = Dataset(self.config)
        self.num_networks = 5
        self.num_samples = 6
        self.num_nodes = 10
        return

    def scenario_1(self):
        networks = {
            i: nx.complete_graph(self.num_nodes) for i in range(self.num_networks)
        }
        inputs = {
            i: np.zeros((self.num_samples, self.num_nodes))
            for i in range(self.num_networks)
        }
        targets = {
            i: np.zeros((self.num_samples, self.num_nodes))
            for i in range(self.num_networks)
        }
        data = {}
        data["networks"] = networks
        data["inputs"] = inputs
        data["targets"] = targets
        self.dataset.data = data

    def scenario_2(self):
        networks = {
            i: nx.complete_graph(self.num_nodes) for i in range(self.num_networks)
        }
        inputs = {
            i: np.random.randint(2, size=(self.num_samples, self.num_nodes))
            for i in range(self.num_networks)
        }
        targets = {
            i: np.random.randint(2, size=(self.num_samples, self.num_nodes))
            for i in range(self.num_networks)
        }
        data = {}
        data["networks"] = networks
        data["inputs"] = inputs
        data["targets"] = targets
        self.dataset.data = data

    def test_get_indices(self):
        self.scenario_1()
        indices = self.dataset.indices
        ref_indices = list(range(self.num_networks * self.num_samples))
        self.assertEqual(self.num_networks * self.num_samples, len(indices))
        self.assertEqual(ref_indices, list(indices.keys()))

    def test_get_weights(self):
        self.scenario_1()
        weights = self.dataset.weights
        ref_weights = {
            i: np.ones((self.num_samples, self.num_nodes))
            for i in range(self.num_networks)
        }
        for i in range(self.num_networks):
            np.testing.assert_array_almost_equal(ref_weights[i], weights[i])

    def test_partition(self):
        self.scenario_1()
        dataset = self.dataset.partition(0.5)
        for i in range(self.num_networks):
            self.assertEqual(self.dataset.networks[i], dataset.networks[i])
            np.testing.assert_array_equal(self.dataset.inputs[i], dataset.inputs[i])
            np.testing.assert_array_equal(self.dataset.targets[i], dataset.targets[i])

            index1 = np.where(self.dataset.weights[i] == 0.0)[0]
            index2 = np.where(dataset.weights[i] > 0.0)[0]
            np.testing.assert_array_equal(index1, index2)
        return

    def test_next(self):
        self.scenario_1()
        it = iter(self.dataset)
        data = next(it)
        i = 0
        for data in self.dataset:
            i += 1
        self.assertEqual(self.num_samples * self.num_networks, i)
        (x, g), y, w = data
        np.testing.assert_array_almost_equal(np.zeros(self.num_nodes), x)
        np.testing.assert_array_almost_equal(np.zeros(self.num_nodes), y)
        np.testing.assert_array_almost_equal(
            np.ones(self.num_nodes) / self.num_nodes, w
        )

    def test_batch(self):
        self.scenario_2()
        batches = self.dataset.to_batch(5)
        i = 0
        for b in batches:
            j = 0
            for bb in b:
                (x, edge_index), y, w = bb
                self.assertEqual(type(x), torch.Tensor)
                self.assertEqual(type(edge_index), torch.Tensor)
                self.assertEqual(type(y), torch.Tensor)
                self.assertEqual(type(w), torch.Tensor)
                j += 1
                pass
            self.assertEqual(5, j)
            i += 1
        self.assertEqual(6, i)
