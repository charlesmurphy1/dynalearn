import networkx as nx
import numpy as np
import torch
import unittest

from dynalearn.nn.models import GenericGNN, GenericWGNN
from dynalearn.config import TrainableConfig


class GenericGNNTest(unittest.TestCase):
    def setUp(self):
        self.in_size = 3
        self.out_size = 3
        self.window_size = 2
        self.nodeattr_size = 2
        self.edgeattr_size = 2
        self.num_nodes = 10
        self.model = GenericGNN(
            self.in_size,
            self.out_size,
            window_size=self.window_size,
            nodeattr_size=self.nodeattr_size,
            edgeattr_size=self.edgeattr_size,
            out_act="softmax",
            normalize=True,
            config=TrainableConfig.sis(),
        )

    def generate_network(self):
        g = nx.gnp_random_graph(self.num_nodes, 0.4)
        for u in g.nodes():
            for i in range(self.nodeattr_size):
                g.nodes[u][f"attr{i}"] = np.random.randn()
        for u, v in g.edges():
            for i in range(self.edgeattr_size):
                g.edges[u, v][f"attr{i}"] = np.random.randn()
        return g

    def test_forward(self):
        g = self.generate_network()
        x = torch.randn(self.num_nodes, self.in_size, self.window_size)
        _x = x
        _g = self.model.transformers["network"].forward(g)
        y = self.model.forward(_x, _g).cpu().detach().numpy()
        self.assertTrue(y.shape == (self.num_nodes, self.out_size))
        np.testing.assert_array_almost_equal(y.sum(-1), np.ones((y.shape[0])))


class GenericWGNNTest(GenericGNNTest):
    def setUp(self):
        self.in_size = 3
        self.out_size = 3
        self.window_size = 2
        self.nodeattr_size = 2
        self.edgeattr_size = 2
        self.num_nodes = 10
        self.model = GenericWGNN(
            self.in_size,
            self.out_size,
            window_size=self.window_size,
            nodeattr_size=self.nodeattr_size,
            edgeattr_size=self.edgeattr_size,
            out_act="softmax",
            normalize=True,
            config=TrainableConfig.sis(),
        )

    def test_forward(self):
        g = self.generate_network()
        x = torch.randn(self.num_nodes, self.in_size, self.window_size)
        _x = x
        _g = self.model.transformers["network"].forward(g)
        y = self.model.forward(_x, _g).cpu().detach().numpy()
        self.assertTrue(y.shape == (self.num_nodes, self.out_size))
        np.testing.assert_array_almost_equal(y.sum(-1), np.ones((y.shape[0])))


if __name__ == "__main__":
    unittest.main()
