import unittest
import networkx as nx
import numpy as np
from dynalearn.dynamics import SIS
from dynalearn.config import DynamicsConfig


class SISTest(unittest.TestCase):
    def setUp(self):
        config = DynamicsConfig.sis_default()
        config.infection = 0.2
        config.recovery = 0.1
        config.initial_infected = 0.5

        self.infection = config.infection
        self.recovery = config.recovery
        self.initial_infected = config.initial_infected

        self.n = 100
        self.g = nx.complete_graph(self.n)
        self.k = np.array(list(dict(self.g.degree()).values()))
        self.x = np.random.randint(2, size=self.n)

        self.model = SIS(config)
        self.model.network = self.g

    def test_change_network(self):
        self.assertEqual(self.model.num_nodes, self.n)
        g = nx.empty_graph(25)
        adj = nx.to_numpy_array(g)
        self.model.network = g
        self.assertEqual(self.model.num_nodes, 25)
        np.testing.assert_array_almost_equal(adj, np.zeros((25, 25)))
        self.model.network = nx.complete_graph(self.n)

    def test_predict_S(self):
        x = np.zeros(self.n)
        ltp = self.model.predict(x)
        ref_ltp = np.zeros((self.n, 2))
        ref_ltp[:, 0] = 1
        np.testing.assert_array_almost_equal(ref_ltp, ltp)

    def test_predict_inf(self):
        x = np.zeros(self.n)
        x[0] = 1
        ltp = self.model.predict(x)
        ref_ltp = np.zeros((self.n, 2))
        ref_ltp[0, 0] = self.recovery
        ref_ltp[0, 1] = 1 - self.recovery
        ref_ltp[1:, 0] = 1 - self.infection
        ref_ltp[1:, 1] = self.infection
        np.testing.assert_array_almost_equal(ref_ltp, ltp)

    def test_predict_rec(self):
        x = np.ones(self.n)
        ltp = self.model.predict(x)
        ref_ltp = np.zeros((self.n, 2))
        ref_ltp[:, 0] = self.recovery
        ref_ltp[:, 1] = 1 - self.recovery
        np.testing.assert_array_almost_equal(ref_ltp, ltp)

    def test_sample(self):
        x = np.zeros(self.n)
        self.assertTrue(np.all(x == self.model.sample(x)))
        x[0] = 1
        self.assertAlmostEqual(
            np.mean(self.model.sample(x)[1:]), self.infection, places=1
        )

    def test_initialstate(self):
        g = nx.empty_graph(1000)
        self.model.network = g
        x = self.model.initial_state()
        self.assertAlmostEqual(np.mean(x), self.initial_infected, places=1)
        self.model.network = self.g
        return


if __name__ == "__main__":
    unittest.main()
