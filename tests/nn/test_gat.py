import networkx as nx
import numpy as np
import torch
import unittest
from dynalearn.nn.models import GraphAttention
from dynalearn.utilities import to_edge_index


class GraphAttentionTest(unittest.TestCase):
    def setUp(self):
        self.num_nodes = 10
        self.in_channels = 8
        self.out_channels = 4
        self.heads = 1
        self.concat = False
        self.bias = False
        self.gat = GraphAttention(
            self.in_channels, self.out_channels, self.heads, self.concat, self.bias
        )

    def test_forward(self):
        x = torch.ones((self.num_nodes, self.in_channels))
        edge_index = to_edge_index(nx.complete_graph(self.num_nodes).to_directed())
        _x = self.gat.linear(x)
        __x = torch.cat([_x, _x], axis=-1)[0].view(1, -1)
        a = torch.sigmoid((__x * self.gat.att_weight.view(1, -1)).sum(-1)).squeeze()
        ref_y = _x + (self.num_nodes - 1) * a * _x

        y = self.gat.forward(x, edge_index)
        torch.testing.assert_allclose(y, ref_y)


if __name__ == "__main__":
    unittest.main()
