import unittest
from dynalearn.networks import BANetwork
import networkx as nx
import numpy as np
from dynalearn.config import NetworkConfig


class BANetworkTest(unittest.TestCase):
    def setUp(self):
        self.n = 100
        self.m = 2
        config = NetworkConfig.barabasialbert(self.n, self.m)
        self.network = BANetwork(config)

    def test_generate(self):
        size = len(self.network.data)
        self.network.generate()
        self.assertEqual(size + 1, len(self.network.data))
        self.network.clear()
        self.assertEqual(0, len(self.network.data))


if __name__ == "__main__":
    unittest.main()
