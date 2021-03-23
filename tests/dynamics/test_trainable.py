import unittest
import networkx as nx
import numpy as np

from dynalearn.dynamics import GNNSEDynamics, GNNDEDynamics
from dynalearn.config import TrainableConfig, NetworkConfig
from torch_geometric.nn.inits import ones
from dynalearn.networks.getter import get as get_network


class GNNSEDynamicsTest(unittest.TestCase):
    def get_model(self):
        self.config = TrainableConfig.sis()
        self.model = GNNSEDynamics(self.config)

    def setUp(self):
        self.model = self.get_model()
        self.num_nodes = 5
        self.num_states = self.config.num_states
        self.window_size = self.config.window_size
        self.network = get_network(NetworkConfig.barabasialbert(self.num_nodes))

    def test_predict(self):
        for i in range(10):
            self.model.nn.reset_parameters()
            self.model.network = self.network.generate(0)
            x = np.random.randint(
                self.model.num_states, size=(self.num_nodes, self.window_size)
            )
            y = self.model.predict(x)
            self.assertFalse(np.any(y == np.nan))
            self.assertEqual(y.shape, (self.num_nodes, self.num_states))
            np.testing.assert_array_almost_equal(y.sum(-1), np.ones(self.num_nodes))

    def test_sample(self):
        self.model.network = self.network.generate(0)
        x = np.random.randint(
            self.model.num_states, size=(self.num_nodes, self.window_size)
        )
        self.model.nn.reset_parameters(ones)
        y = self.model.sample(x)
        self.assertEqual(y.shape, (self.num_nodes,))
        self.model.nn.reset_parameters()


class GNNDEDynamicsTest(unittest.TestCase):
    def get_model(self):
        self.config = TrainableConfig.dsir()
        self.model = GNNDEDynamics(self.config)


if __name__ == "__main__":
    unittest.main()
