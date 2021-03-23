import unittest
import numpy as np
import networkx as nx
import torch

from dynalearn.nn.models import UnivariateMPL, UnivariateRNN
from dynalearn.config import TrainableConfig, NetworkConfig
from dynalearn.networks.getter import get as get_network


class UnivariateMPLTest(unittest.TestCase):
    def setUp(self):
        self.in_size = 3
        self.out_size = 3
        self.window_size = 2
        self.nodeattr_size = 0
        self.edgeattr_size = 0
        self.num_nodes = 10
        self.model = UnivariateMPL(
            self.in_size,
            self.out_size,
            window_size=self.window_size,
            nodeattr_size=self.nodeattr_size,
            out_act="softmax",
            normalize=True,
            config=TrainableConfig.sis_uv(),
        )
        self.network = get_network(NetworkConfig.barabasialbert(self.num_nodes))

    def test_forward(self):
        g = self.network.generate(0)
        x = torch.randn(self.num_nodes, self.in_size, self.window_size)
        y = torch.randn(self.num_nodes, self.in_size)
        w = torch.randn(self.num_nodes)
        data = (x, g), y, w
        (x, g), y, w = self.model.transformers.forward(data)
        y = self.model.forward(x, g).cpu().detach().numpy()
        self.assertTrue(y.shape == (self.num_nodes, self.out_size))
        np.testing.assert_array_almost_equal(y.sum(-1), np.ones((y.shape[0])))


class UnivariateRNNTest(unittest.TestCase):
    def setUp(self):
        self.in_size = 3
        self.out_size = 3
        self.window_size = 2
        self.nodeattr_size = 0
        self.edgeattr_size = 0
        self.num_nodes = 10
        self.rnn = "LSTM"
        self.model = UnivariateRNN(
            self.in_size,
            self.out_size,
            rnn=self.rnn,
            window_size=self.window_size,
            nodeattr_size=self.nodeattr_size,
            out_act="softmax",
            normalize=True,
            config=TrainableConfig.sis_uv(),
        )
        self.network = get_network(NetworkConfig.barabasialbert(self.num_nodes))

    def test_forward(self):
        g = self.network.generate(0)
        x = torch.randn(self.num_nodes, self.in_size, self.window_size)
        y = torch.randn(self.num_nodes, self.in_size)
        w = torch.randn(self.num_nodes)
        data = (x, g), y, w
        (x, g), y, w = self.model.transformers.forward(data)
        y = self.model.forward(x, g).cpu().detach().numpy()
        self.assertTrue(y.shape == (self.num_nodes, self.out_size))
        np.testing.assert_array_almost_equal(y.sum(-1), np.ones((y.shape[0])))


if __name__ == "__main__":
    unittest.main()
