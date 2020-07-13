import networkx as nx
import numpy as np
import torch

from dynalearn.config import DatasetConfig, DynamicsConfig
from dynalearn.datasets import DiscreteDataset, Sampler, Data, NetworkData, WindowedData
from dynalearn.dynamics import SIS
from unittest import TestCase


class DiscreteDatasetTest(TestCase):
    def setUp(self):
        self.config = DatasetConfig.plain_discrete_default()
        self.num_states = 2
        self.num_networks = 5
        self.num_samples = 6
        self.num_nodes = 10
        self.window_size = 2
        self.window_step = 2
        self.config.window_size = self.window_size
        self.config.window_step = self.window_step
        self.dataset = DiscreteDataset(self.config)
        self.dataset.m_dynamics = SIS(DynamicsConfig.sis_default())
        return

    def scenario_1(self):
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

    def scenario_2(self):
        networks = NetworkData(
            data=[nx.complete_graph(self.num_nodes) for i in range(self.num_networks)]
        )
        inputs = {
            i: WindowedData(
                data=np.random.randint(
                    self.num_states, size=(self.num_samples, self.num_nodes)
                ),
                window_size=self.window_size,
                window_step=self.window_step,
            )
            for i in range(self.num_networks)
        }
        targets = {
            i: Data(
                data=np.random.randint(
                    self.num_states, size=(self.num_samples, self.num_nodes)
                )
            )
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
            np.testing.assert_array_equal(self.dataset.networks[i], dataset.networks[i])
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
        x_ref = np.zeros((self.window_size, self.num_nodes))
        y_ref = np.zeros((self.num_nodes, self.num_states))
        y_ref[:, 0] = 1
        w_ref = np.ones(self.num_nodes) / self.num_nodes
        np.testing.assert_array_almost_equal(x_ref, x)
        np.testing.assert_array_almost_equal(y_ref, y)
        np.testing.assert_array_almost_equal(w_ref, w)

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
