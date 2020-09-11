import unittest
import networkx as nx
import numpy as np

from dynalearn.dynamics import SISSIS
from dynalearn.config import DynamicsConfig


class SISSISTest(unittest.TestCase):
    def setUp(self):
        self.config = DynamicsConfig.sissis()

        self.n = 100
        self.g = nx.complete_graph(self.n)
        self.k = np.array(list(dict(self.g.degree()).values()))
        self.x = np.random.randint(2, size=self.n)

        self.model = SISSIS(self.config)
        self.model.network = self.g

    def test_predict(self):
        x = np.zeros(self.n)
        ltp = self.model.predict(x)
        ref_ltp = np.zeros((self.n, 4))
        ref_ltp[:, 0] = 1
        np.testing.assert_array_almost_equal(ltp, ref_ltp)

        x = np.zeros(self.n)
        x[0] = 1
        ltp = self.model.predict(x)
        ref_ltp = np.zeros((self.n, 4))
        ref_ltp[0, 0] = self.config.recovery1
        ref_ltp[0, 1] = 1 - self.config.recovery1
        ref_ltp[1:, 0] = 1 - self.config.infection1
        ref_ltp[1:, 1] = self.config.infection1
        np.testing.assert_array_almost_equal(ltp, ref_ltp)

        x = np.zeros(self.n)
        x[0] = 2
        ltp = self.model.predict(x)
        ref_ltp = np.zeros((self.n, 4))
        ref_ltp[0, 0] = self.config.recovery2
        ref_ltp[0, 2] = 1 - self.config.recovery2
        ref_ltp[1:, 0] = 1 - self.config.infection2
        ref_ltp[1:, 2] = self.config.infection2
        np.testing.assert_array_almost_equal(ltp, ref_ltp)

        x = np.zeros(self.n)
        x[0] = 3
        ltp = self.model.predict(x)
        p1, p2 = self.config.infection1, self.config.infection2
        q1, q2 = self.config.recovery1, self.config.recovery2
        c = self.config.coupling
        ref_ltp = np.zeros((self.n, 4))
        ref_ltp[0, 0] = (q1) * (q2)
        ref_ltp[0, 1] = (1 - q1) * (q2)
        ref_ltp[0, 2] = (q1) * (1 - q2)
        ref_ltp[0, 3] = (1 - q1) * (1 - q2)
        ref_ltp[1:, 0] = (1 - c * p1) * (1 - c * p2)
        ref_ltp[1:, 1] = (c * p1) * (1 - c * p2)
        ref_ltp[1:, 2] = (1 - c * p1) * (c * p2)
        ref_ltp[1:, 3] = (c * p1) * (c * p2)
        np.testing.assert_array_almost_equal(ltp, ref_ltp)

        x = np.ones(self.n) * 1
        ltp = self.model.predict(x)
        ref_ltp = np.zeros((self.n, 4))
        ref_ltp[:, 0] = self.config.recovery1
        ref_ltp[:, 1] = 1 - self.config.recovery1
        np.testing.assert_array_almost_equal(ltp, ref_ltp)

        x = np.ones(self.n) * 2
        ltp = self.model.predict(x)
        ref_ltp = np.zeros((self.n, 4))
        ref_ltp[:, 0] = self.config.recovery2
        ref_ltp[:, 2] = 1 - self.config.recovery2
        np.testing.assert_array_almost_equal(ltp, ref_ltp)

        x = np.ones(self.n) * 3
        ltp = self.model.predict(x)
        ref_ltp = np.zeros((self.n, 4))
        ref_ltp[:, 0] = (self.config.recovery1) * (self.config.recovery2)
        ref_ltp[:, 1] = (1 - self.config.recovery1) * (self.config.recovery2)
        ref_ltp[:, 2] = (self.config.recovery1) * (1 - self.config.recovery2)
        ref_ltp[:, 3] = (1 - self.config.recovery1) * (1 - self.config.recovery2)
        np.testing.assert_array_almost_equal(ltp, ref_ltp)

    def test_initialstate(self):
        g = nx.empty_graph(1000)
        self.model.network = g
        x = self.model.initial_state(0.5)
        self.assertAlmostEqual(np.mean(x == 0), 0.5, places=1)
        self.assertAlmostEqual(np.mean(x == 1), np.mean(x == 2), places=1)
        self.model.network = self.g
        return


if __name__ == "__main__":
    unittest.main()
