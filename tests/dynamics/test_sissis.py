import unittest
from dynalearn.dynamics import SISSIS, DynamicsConfig
import networkx as nx
import numpy as np


class SISSISTest(unittest.TestCase):
    def setUp(self):
        config = DynamicsConfig.sissis_default()
        config.infection1 = 0.5
        config.infection2 = 0.6
        config.recovery1 = 0.2
        config.recovery2 = 0.3
        config.coupling = 0.1
        config.initial_infected = 0.5

        self.infection1 = config.infection1
        self.infection2 = config.infection2
        self.recovery1 = config.recovery1
        self.recovery2 = config.recovery2
        self.coupling = config.coupling
        self.initial_infected = config.initial_infected

        self.n = 100
        self.g = nx.complete_graph(self.n)
        self.k = np.array(list(dict(self.g.degree()).values()))
        self.x = np.random.randint(2, size=self.n)

        self.model = SISSIS(config)
        self.model.network = self.g

    def test_predict_S(self):
        x = np.zeros(self.n)
        ltp = self.model.predict(x)
        ref_ltp = np.zeros((self.n, 4))
        ref_ltp[:, 0] = 1
        self.assertTrue(np.all(abs(ref_ltp - ltp) < 1e-10))

    def test_predict_infIS(self):
        x = np.zeros(self.n)
        x[0] = 1
        ltp = self.model.predict(x)
        ref_ltp = np.zeros((self.n, 4))
        ref_ltp[0, 0] = self.recovery1
        ref_ltp[0, 1] = 1 - self.recovery1
        ref_ltp[1:, 0] = 1 - self.infection1
        ref_ltp[1:, 1] = self.infection1
        self.assertTrue(np.all(abs(ref_ltp - ltp) < 1e-10))

    def test_predict_infSI(self):
        x = np.zeros(self.n)
        x[0] = 2
        ltp = self.model.predict(x)
        ref_ltp = np.zeros((self.n, 4))
        ref_ltp[0, 0] = self.recovery2
        ref_ltp[0, 2] = 1 - self.recovery2
        ref_ltp[1:, 0] = 1 - self.infection2
        ref_ltp[1:, 2] = self.infection2
        self.assertTrue(np.all(abs(ref_ltp - ltp) < 1e-10))

    def test_predict_infII(self):
        x = np.zeros(self.n)
        x[0] = 3
        ltp = self.model.predict(x)
        p1, p2 = self.infection1, self.infection2
        q1, q2 = self.recovery1, self.recovery2
        c = self.coupling
        ref_ltp = np.zeros((self.n, 4))
        ref_ltp[0, 0] = (q1) * (q2)
        ref_ltp[0, 1] = (1 - q1) * (q2)
        ref_ltp[0, 2] = (q1) * (1 - q2)
        ref_ltp[0, 3] = (1 - q1) * (1 - q2)
        ref_ltp[1:, 0] = (1 - c * p1) * (1 - c * p2)
        ref_ltp[1:, 1] = (c * p1) * (1 - c * p2)
        ref_ltp[1:, 2] = (1 - c * p1) * (c * p2)
        ref_ltp[1:, 3] = (c * p1) * (c * p2)
        self.assertTrue(np.all(abs(ref_ltp - ltp) < 1e-10))

    def test_predict_recIS(self):
        x = np.ones(self.n) * 1
        ltp = self.model.predict(x)
        ref_ltp = np.zeros((self.n, 4))
        ref_ltp[:, 0] = self.recovery1
        ref_ltp[:, 1] = 1 - self.recovery1
        self.assertTrue(np.all(abs(ref_ltp - ltp) < 1e-10))

    def test_predict_recSI(self):
        x = np.ones(self.n) * 2
        ltp = self.model.predict(x)
        ref_ltp = np.zeros((self.n, 4))
        ref_ltp[:, 0] = self.recovery2
        ref_ltp[:, 2] = 1 - self.recovery2
        self.assertTrue(np.all(abs(ref_ltp - ltp) < 1e-10))

    def test_predict_recII(self):
        x = np.ones(self.n) * 3
        ltp = self.model.predict(x)
        ref_ltp = np.zeros((self.n, 4))
        ref_ltp[:, 0] = (self.recovery1) * (self.recovery2)
        ref_ltp[:, 1] = (1 - self.recovery1) * (self.recovery2)
        ref_ltp[:, 2] = (self.recovery1) * (1 - self.recovery2)
        ref_ltp[:, 3] = (1 - self.recovery1) * (1 - self.recovery2)
        self.assertTrue(np.all(abs(ref_ltp - ltp) < 1e-10))

    def test_initialstate(self):
        g = nx.empty_graph(1000)
        self.model.network = g
        x = self.model.initial_state()
        self.assertAlmostEqual(np.mean(x == 0), 1 - self.initial_infected, places=1)
        self.assertAlmostEqual(np.mean(x == 1), np.mean(x == 2), places=1)
        self.model.network = self.g
        return


if __name__ == "__main__":
    unittest.main()
