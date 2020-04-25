import unittest
from dynalearn.networks import ERNetwork, NetworkConfig
import networkx as nx
import numpy as np


class ERNetworkTest(unittest.TestCase):
    def setUp(self):
        self.p = 0.5
        self.n = 100
        config = NetworkConfig.erdosrenyi(self.n, self.p)
        self.network = ERNetwork(config)

    def test_generate(self):
        size = len(self.network.data)
        self.network.generate()
        self.assertEqual(size + 1, len(self.network.data))
        self.network.clear()
        self.assertEqual(0, len(self.network.data))

    def test_density(self):
        self.network.generate()
        g = self.network.data[0]
        density = 2 * g.number_of_edges() / (self.n - 1) / self.n
        self.assertTrue(np.abs(self.p - density) < 5.0 / self.n)


if __name__ == "__main__":
    unittest.main()
