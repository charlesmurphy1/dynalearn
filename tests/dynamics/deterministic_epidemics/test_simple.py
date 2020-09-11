import networkx as nx
import numpy as np
import torch
import unittest

from dynalearn.dynamics import DSIS, DSIR
from dynalearn.config import DynamicsConfig


class DSISTest(unittest.TestCase):
    def setUp(self):
        self.config = DynamicsConfig.dsis()

        self.n = 100
        self.g = nx.complete_graph(self.n)
        self.k = np.array(list(dict(self.g.degree()).values()))

        self.model = DSIS(self.config)
        self.model.network = self.g

    def test_predict(self):
        x = self.model.initial_state()
        y = self.model.predict(x)
        np.testing.assert_array_almost_equal(y.sum(-1), np.ones(self.n))

class DSIRTest(unittest.TestCase):
    def setUp(self):
        self.config = DynamicsConfig.dsir()

        self.n = 100
        self.g = nx.complete_graph(self.n)
        self.k = np.array(list(dict(self.g.degree()).values()))

        self.model = DSIR(self.config)
        self.model.network = self.g

    def test_predict(self):
        x = self.model.initial_state()
        y = self.model.predict(x)
        np.testing.assert_array_almost_equal(y.sum(-1), np.ones(self.n))
