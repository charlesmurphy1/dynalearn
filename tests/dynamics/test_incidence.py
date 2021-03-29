import networkx as nx
import numpy as np
import torch
import time
import unittest

from dynalearn.dynamics import IncidenceSIR
from dynalearn.config import DynamicsConfig, NetworkConfig
from dynalearn.networks.getter import get as get_network


class IncidenceSIRTest(unittest.TestCase):
    def setUp(self):
        self.num_nodes = 100
        self.model = IncidenceSIR(DynamicsConfig.incsir())
        self.num_states = self.model.num_states
        self.lag = self.model.lag
        self.network = get_network(NetworkConfig.barabasialbert(self.num_nodes, 2))
        self.model.network = self.network.generate(int(time.time()))

    def test_change_network(self):
        self.assertEqual(self.model.num_nodes, self.num_nodes)
        self.model.network = self.network.generate(int(time.time()))
        self.assertEqual(self.model.num_nodes, self.num_nodes)

    def test_predict(self):
        x = np.random.poisson(50, size=self.num_nodes)
        y = self.model.predict(x)
        self.assertFalse(np.any(y == np.nan))
        self.assertEqual(y.shape, (self.num_nodes, self.num_states))

    def test_sample(self):
        x = self.model.initial_state()
        y = self.model.sample(x)
        self.assertTrue(np.any(x != y))

    def test_initialstate(self):
        x = self.model.initial_state(squeeze=False)
        self.assertEqual(x.shape, (self.num_nodes, self.num_states, self.lag))
