import networkx as nx
import numpy as np
import torch
import unittest
from dynalearn.nn.models import GraphAttention
from dynalearn.utilities import to_edge_index


class GraphAttentionTest(unittest.TestCase):
    def setUp(self):
        self.num_nodes = 10
        self.p = 0.5
        self.in_channels = 8
        self.out_channels = 4
        self.heads = 7
        self.concat = True
        self.bias = True
        self.attn_bias = True
        self.edge_in_channels = 0
        self.edge_out_channels = 4
        self.self_attention = True
        self.gat = GraphAttention(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            heads=self.heads,
            concat=self.concat,
            bias=self.bias,
            attn_bias=self.attn_bias,
            edge_in_channels=self.edge_in_channels,
            edge_out_channels=self.edge_out_channels,
            self_attention=self.self_attention,
        )

    def test_forward(self):
        x = torch.ones((self.num_nodes, self.in_channels))
        g = nx.gnp_random_graph(self.num_nodes, self.p).to_directed()
        num_edges = g.number_of_edges()
        edge_index = to_edge_index(g)
        edge_attr = torch.rand(edge_index.size(1), self.edge_in_channels)
        out = self.gat.forward(x, edge_index, edge_attr=edge_attr)
        if self.concat:
            c1 = self.heads * self.out_channels
            c2 = self.heads * self.edge_out_channels
        else:
            c1 = self.out_channels
            c2 = self.edge_out_channels
        if isinstance(out, tuple):
            x, edge_attr = out
            self.assertEqual(x.shape, torch.Size([self.num_nodes, c1]))
            self.assertEqual(edge_attr.shape, torch.Size([num_edges, c2]))
        else:
            self.assertEqual(out.shape, torch.Size([self.num_nodes, c1]))


if __name__ == "__main__":
    unittest.main()
