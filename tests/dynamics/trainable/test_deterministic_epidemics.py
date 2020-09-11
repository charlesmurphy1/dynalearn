import unittest
import networkx as nx
import numpy as np

from dynalearn.dynamics import TrainableDeterministicEpidemics
from dynalearn.config import TrainableConfig
from torch_geometric.nn.inits import ones


class TrainableDeterministicEpidemicsTest(unittest.TestCase):
    def setUp(self):
        self.config = TrainableConfig.dsis()
        self.model = TrainableDeterministicEpidemics(self.config)
        self.num_nodes = 5
        self.num_states = self.config.num_states
        self.window_size = self.config.window_size
        self.model.network = nx.barabasi_albert_graph(self.num_nodes, 2)

    def test_predict(self):
        for i in range(10):
            self.model.nn.reset_parameters()
            x = self.model.initial_state()
            y = self.model.predict(x)
            self.assertFalse(np.any(y == np.nan))

    def test_sample(self):
        x = self.model.initial_state()
        self.model.nn.reset_parameters(ones)
        y = self.model.sample(x)
        self.assertEqual(y.shape, (self.num_nodes, self.num_states))
        self.model.nn.reset_parameters()


if __name__ == "__main__":
    unittest.main()
