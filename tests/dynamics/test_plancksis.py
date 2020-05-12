import unittest
import networkx as nx
import numpy as np

from dynalearn.dynamics import PlanckSIS
from dynalearn.dynamics.activation import planck
from dynalearn.config import DynamicsConfig


class PlanckSISTest(unittest.TestCase):
    def setUp(self):
        config = DynamicsConfig.plancksis_default()
        config.temperature = 1
        config.recovery = 0.1
        config.initial_infected = 0.5

        self.temperature = config.temperature
        self.recovery = config.recovery
        self.initial_infected = config.initial_infected

        self.n = 100
        self.g = nx.complete_graph(self.n)
        self.k = np.array(list(dict(self.g.degree()).values()))
        self.x = np.random.randint(2, size=self.n)

        self.model = PlanckSIS(config)
        self.model.network = self.g

    def test_predict_S(self):
        x = np.zeros(self.n)
        ltp = self.model.predict(x)
        ref_ltp = np.zeros((self.n, 2))
        ref_ltp[:, 0] = 1
        np.testing.assert_array_almost_equal(ltp, ref_ltp)

    def test_predict_inf(self):
        x = np.zeros(self.n)
        x[0] = 1
        ltp = self.model.predict(x)
        ref_ltp = np.zeros((self.n, 2))
        ref_ltp[0, 0] = self.recovery
        ref_ltp[0, 1] = 1 - self.recovery
        ref_ltp[1:, 0] = 1 - planck(np.ones(self.n - 1), self.temperature)
        ref_ltp[1:, 1] = planck(np.ones(self.n - 1), self.temperature)
        np.testing.assert_array_almost_equal(ltp, ref_ltp)

    def test_predict_rec(self):
        x = np.ones(self.n)
        ltp = self.model.predict(x)
        ref_ltp = np.zeros((self.n, 2))
        ref_ltp[:, 0] = self.recovery
        ref_ltp[:, 1] = 1 - self.recovery
        np.testing.assert_array_almost_equal(ltp, ref_ltp)


if __name__ == "__main__":
    unittest.main()
