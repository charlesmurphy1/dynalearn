import unittest
import networkx as nx
import numpy as np

from dynalearn.dynamics import LearnableEpidemics
from dynalearn.config import DynamicsConfig
from torch_geometric.nn.inits import ones


class LearnableEpidemicsTest(unittest.TestCase):
    def setUp(self):
        self.config = DynamicsConfig.sissis_gnn_default()
        self.model = LearnableEpidemics(self.config)
        self.num_nodes = 5
        self.num_states = self.config.num_states

    def scenario_1(self):
        self.model.network = nx.empty_graph(self.num_nodes)
        x = np.ones((self.num_nodes, 1))
        return x

    def scenario_2(self):
        self.model.network = nx.barabasi_albert_graph(self.num_nodes, 1)
        x = np.random.randint(self.model.num_states, size=self.num_nodes)
        return x

    def test_predict(self):
        x = self.scenario_1()
        self.model.nn.reset_parameter(ones)
        y = self.model.predict(x)
        self.assertEqual(y.shape, (self.num_nodes, self.num_states))
        np.testing.assert_array_almost_equal(y.sum(-1), np.ones(self.num_nodes))
        self.assertFalse(np.any(y == np.nan))
        self.model.nn.reset_parameter()

    def test_predict_random(self):
        for i in range(10):
            self.model.nn.reset_parameter()
            x = self.scenario_2()
            y = self.model.predict(x)
            if np.any(y == np.nan):
                print(x)

    def test_sample(self):
        x = self.scenario_1()
        self.model.nn.reset_parameter(ones)
        y = self.model.sample(x)
        self.assertEqual(y.shape, (self.num_nodes,))
        self.model.nn.reset_parameter()


if __name__ == "__main__":
    unittest.main()
