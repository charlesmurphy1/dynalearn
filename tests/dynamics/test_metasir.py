import unittest
import networkx as nx
import numpy as np
from dynalearn.dynamics import MetaSIR
from dynalearn.config import DynamicsConfig


class MetaSIRTest(unittest.TestCase):
    def setUp(self):
        config = DynamicsConfig.metasir_default()
        config.infection_prob = 0.04
        config.recovery_prob = 0.08
        config.infection_type = 2
        config.diffusion_susceptible = 0.1
        config.diffusion_infected = 0.1
        config.diffusion_recovered = 0.1
        self.num_states = 3
        config.density = 1000

        self.infection_prob = config.infection_prob
        self.recovery_prob = config.recovery_prob
        self.infection_type = config.infection_type
        self.diffusion_susceptible = config.diffusion_susceptible
        self.diffusion_infected = config.diffusion_infected
        self.diffusion_recovered = config.diffusion_recovered
        self.state_dist = config.state_dist
        self.density = config.density

        self.n = 10
        self.g = nx.gnp_random_graph(self.n, 0.1)
        self.k = np.array(list(dict(self.g.degree()).values()))
        self.x = np.random.randint(self.num_states, size=self.n)

        self.model = MetaSIR(config)
        self.model.network = self.g

    def test_reaction(self):
        x = np.zeros([self.model.num_nodes, self.num_states])
        x[:, 0] = 1
        ltp = self.model.reaction(x)
        ref_ltp = np.zeros([self.model.num_nodes, self.num_states, self.num_states])
        ref_ltp[:, 0, 0] = 1
        ref_ltp[:, 0, 1] = 0
        ref_ltp[:, 0, 2] = 0
        ref_ltp[:, 1, 0] = 0
        ref_ltp[:, 1, 1] = 1 - self.recovery_prob
        ref_ltp[:, 1, 2] = self.recovery_prob
        ref_ltp[:, 2, 0] = 0
        ref_ltp[:, 2, 1] = 0
        ref_ltp[:, 2, 2] = 1
        np.testing.assert_array_almost_equal(ref_ltp, ltp)

        x[:, 0] = 0
        x[:, 1] = 1
        n = x.sum(-1)
        ltp = self.model.reaction(x)
        ref_ltp[:, 0, 0] = (1 - self.infection_prob / n) ** x[:, 1]
        ref_ltp[:, 0, 1] = 1 - (1 - self.infection_prob / n) ** x[:, 1]
        np.testing.assert_array_almost_equal(ref_ltp, ltp)

    def test_diffusion(self):
        g = nx.complete_graph(self.model.num_nodes)
        self.model.network = g
        x = self.model.initial_state()
        ref_ltp = np.array(
            [
                self.diffusion_susceptible,
                self.diffusion_infected,
                self.diffusion_recovered,
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
        g = nx.empty_graph(100)
        self.model.network = g
        x = self.model.initial_state()
        n = x.sum(-1)
        self.assertAlmostEqual(
            np.mean(n) / self.density, 1, places=1,
        )
        self.model.network = self.g


if __name__ == "__main__":
    unittest.main()
