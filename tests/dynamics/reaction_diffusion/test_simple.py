import unittest
import networkx as nx
import numpy as np
from dynalearn.dynamics import RDSIS, RDSIR
from dynalearn.config import DynamicsConfig


class RDSISTest(unittest.TestCase):
    def setUp(self):
        self.config = DynamicsConfig.rdsis()

        self.n = 10
        self.g = nx.gnp_random_graph(self.n, 0.1)
        self.k = np.array(list(dict(self.g.degree()).values()))

        self.model = RDSIS(self.config)
        self.model.network = self.g

    def test_reaction(self):
        x = np.zeros([self.model.num_nodes, self.model.num_states])
        x[:, 0] = 1
        ltp = self.model.reaction(x)
        ref_ltp = np.zeros(
            [self.model.num_nodes, self.model.num_states, self.model.num_states]
        )
        ref_ltp[:, 0, 0] = 1
        ref_ltp[:, 0, 1] = 0
        ref_ltp[:, 1, 0] = self.config.recovery_prob
        ref_ltp[:, 1, 1] = 1 - self.config.recovery_prob
        np.testing.assert_array_almost_equal(ref_ltp, ltp)

        x[:, 0] = 0
        x[:, 1] = 1
        n = x.sum(-1)
        ltp = self.model.reaction(x)
        ref_ltp[:, 0, 0] = (1 - self.config.infection_prob / n) ** x[:, 1]
        ref_ltp[:, 0, 1] = 1 - (1 - self.config.infection_prob / n) ** x[:, 1]
        np.testing.assert_array_almost_equal(ref_ltp, ltp)

    def test_diffusion(self):
        g = nx.complete_graph(self.model.num_nodes)
        self.model.network = g
        x = self.model.initial_state()
        ref_ltp = np.array(
            [self.config.diffusion_susceptible, self.config.diffusion_infected,]
        )
        ltp = self.model.diffusion(x)
        k = dict(g.degree())
        for (i, j) in g.edges():
            np.testing.assert_array_almost_equal(ref_ltp / k[j], ltp[i, j])
        self.model.network = self.g

    def test_predict(self):
        x = self.model.initial_state()
        y = self.model.predict(x)

    def test_sample(self):
        x = self.model.initial_state()
        y = self.model.sample(x)

    def test_initialstate(self):
        self.model.initial_state()


class RDSIRTest(unittest.TestCase):
    def setUp(self):
        self.config = DynamicsConfig.rdsir()

        self.n = 10
        self.g = nx.gnp_random_graph(self.n, 0.1)
        self.k = np.array(list(dict(self.g.degree()).values()))

        self.model = RDSIR(self.config)
        self.model.network = self.g

    def test_reaction(self):
        x = np.zeros([self.model.num_nodes, self.model.num_states])
        x[:, 0] = 1
        ltp = self.model.reaction(x)
        ref_ltp = np.zeros(
            [self.model.num_nodes, self.model.num_states, self.model.num_states]
        )
        ref_ltp[:, 0, 0] = 1
        ref_ltp[:, 0, 1] = 0
        ref_ltp[:, 0, 2] = 0
        ref_ltp[:, 1, 0] = 0
        ref_ltp[:, 1, 1] = 1 - self.config.recovery_prob
        ref_ltp[:, 1, 2] = self.config.recovery_prob
        ref_ltp[:, 2, 0] = 0
        ref_ltp[:, 2, 1] = 0
        ref_ltp[:, 2, 2] = 1
        np.testing.assert_array_almost_equal(ref_ltp, ltp)

        x[:, 0] = 0
        x[:, 1] = 1
        n = x.sum(-1)
        ltp = self.model.reaction(x)
        ref_ltp[:, 0, 0] = (1 - self.config.infection_prob / n) ** x[:, 1]
        ref_ltp[:, 0, 1] = 1 - (1 - self.config.infection_prob / n) ** x[:, 1]
        np.testing.assert_array_almost_equal(ref_ltp, ltp)

    def test_diffusion(self):
        g = nx.complete_graph(self.model.num_nodes)
        self.model.network = g
        x = self.model.initial_state()
        ref_ltp = np.array(
            [
                self.config.diffusion_susceptible,
                self.config.diffusion_infected,
                self.config.diffusion_recovered,
            ]
        )
        ltp = self.model.diffusion(x)
        k = dict(g.degree())
        for (i, j) in g.edges():
            np.testing.assert_array_almost_equal(ref_ltp / k[j], ltp[i, j])
        self.model.network = self.g

    def test_predict(self):
        x = self.model.initial_state()
        y = self.model.predict(x)

    def test_sample(self):
        x = self.model.initial_state()
        y = self.model.sample(x)

    def test_initialstate(self):
        self.model.initial_state()


if __name__ == "__main__":
    unittest.main()
