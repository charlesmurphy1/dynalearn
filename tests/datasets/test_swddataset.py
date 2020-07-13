import networkx as nx
import numpy as np

from dynalearn.config import DatasetConfig, DynamicsConfig
from dynalearn.datasets import (
    StateWeightedDiscreteDataset,
    Sampler,
    Data,
    NetworkData,
    WindowedData,
)
from dynalearn.dynamics import SIS

from unittest import TestCase


class StateWeightedDatasetTest(TestCase):
    def setUp(self):
        self.config = DatasetConfig.state_weighted_discrete_default()
        self.num_states = 2
        self.num_networks = 5
        self.num_samples = 6
        self.num_nodes = 10
        self.window_size = 2
        self.window_step = 1
        self.config.window_size = self.window_size
        self.config.window_step = self.window_step
        self.dataset = StateWeightedDiscreteDataset(self.config)
        self.dataset.m_dynamics = SIS(DynamicsConfig.sis_default())
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
        back_step = (self.window_size - 1) * self.window_step
        x = self.num_networks * self.num_nodes * (self.num_samples - back_step)

        for i in range(self.num_networks):
            ref_weights = np.ones((self.num_samples, self.num_nodes))
            ref_weights[back_step:] = x
            np.testing.assert_array_equal(ref_weights, weights[i])

        self.scenario_2()
        weights = self.dataset._get_weights_()

        for i in range(self.num_networks):
            ref_weights = np.ones((self.num_samples, self.num_nodes))
            ref_weights[back_step:] = x
            np.testing.assert_array_equal(ref_weights, weights[i])
