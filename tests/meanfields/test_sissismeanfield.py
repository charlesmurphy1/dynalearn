import numpy as np
import unittest

from dynalearn.dynamics import SISSIS
from dynalearn.meanfields import SISSISMeanfield, GenericMeanfield
from dynalearn.utilities import (
    kronecker_distribution,
    poisson_distribution,
    all_combinations,
    numba_all_combinations,
)
from itertools import product
from scipy.special import binom
from dynalearn.config import DynamicsConfig


class TestMeanfield(unittest.TestCase):
    def setUp(self):
        self.config = DynamicsConfig.sissis_default()
        self.k = 5
        self.n_k = 5
        self.p_k = kronecker_distribution(self.k)
        self.model = SISSIS(self.config)
        self.ref_mf = SISSISMeanfield(self.p_k, self.model)
        self.mf = GenericMeanfield(self.p_k, self.model, True)

    def scenario_uni1(self):
        self.p_k = kronecker_distribution(self.k)
        self.ref_mf.p_k = self.p_k
        self.mf.p_k = self.p_k
        self.x = {k: np.ones(self.model.num_states) for k in self.ref_mf.k}
        self.x = self.ref_mf.normalize_state(self.x)

    def scenario_uni2(self):
        self.p_k = poisson_distribution(self.k, self.n_k)
        self.ref_mf.p_k = self.p_k
        self.mf.p_k = self.p_k
        self.x = {k: np.ones(self.model.num_states) for k in self.ref_mf.k}
        self.x = self.ref_mf.normalize_state(self.x)

    def scenario_random(self):
        self.x = self.mf.random_state()

    def test_ltp(self):
        for i in range(50):
            l1 = np.random.randint(self.k)
            l2 = np.random.randint(self.k - l1)
            l3 = np.random.randint(self.k - l1 - l2)
            ns = (self.k - l1 - l2 - l3, l1, l2, l3)
            p1 = lambda l: (1 - self.model.infection1) ** l
            p2 = lambda l: (1 - self.model.infection2) ** l
            q1 = lambda l: (1 - self.model.coupling * self.model.infection1) ** l
            q2 = lambda l: (1 - self.model.coupling * self.model.infection2) ** l
            r1 = self.model.recovery1
            r2 = self.model.recovery2
            ref_ltp = np.array(
                [
                    [
                        (p1(l1) * q1(l3)) * (p2(l2) * q2(l3)),
                        (1 - p1(l1) * q1(l3)) * (p2(l2) * q2(l3)),
                        (p1(l1) * q1(l3)) * (1 - p2(l2) * q2(l3)),
                        (1 - p1(l1) * q1(l3)) * (1 - p2(l2) * q2(l3)),
                    ],
                    [
                        (r1) * (q2(l2) * q2(l3)),
                        (1 - r1) * (q2(l2) * q2(l3)),
                        (r1) * (1 - q2(l2) * q2(l3)),
                        (1 - r1) * (1 - q2(l2) * q2(l3)),
                    ],
                    [
                        (q1(l1) * q1(l3)) * (r2),
                        (1 - q1(l1) * q1(l3)) * (r2),
                        (q1(l1) * q1(l3)) * (1 - r2),
                        (1 - q1(l1) * q1(l3)) * (1 - r2),
                    ],
                    [
                        (r1) * (r2),
                        (1 - r1) * (r2),
                        (r1) * (1 - r2),
                        (1 - r1) * (1 - r2),
                    ],
                ]
            )
            neighbor_states = np.array(all_combinations(self.k, 4))
            index = np.where(np.prod(ns == neighbor_states, axis=-1))[0][0]
            ltp = self.mf.ltp[self.k][index]
            np.testing.assert_array_almost_equal(ref_ltp, ltp)

    def test_update(self):
        self.scenario_uni1()
        ref_x = self.ref_mf.update(self.x)
        x = self.mf.update(self.x)
        for k in self.x:
            np.testing.assert_array_almost_equal(ref_x[k], x[k])

        self.scenario_uni2()
        ref_x = self.ref_mf.update(self.x)
        x = self.mf.update(self.x)
        for k in self.x:
            np.testing.assert_array_almost_equal(ref_x[k], x[k])

    def test_marginal_ltp(self):
        self.scenario_uni1()
        for i, j in product(range(self.mf.num_states), range(self.mf.num_states)):
            ref_phi = self.ref_mf.phi(self.x)
            ref_mltp = self.ref_mf.marginal_ltp(i, j, self.k, ref_phi)

            phi = self.mf.phi(self.x)
            mltp = self.mf.marginal_ltp(i, j, self.k, phi)
            self.assertAlmostEqual(ref_mltp, mltp, places=6)

        self.scenario_uni2()
        for i, j in product(range(self.mf.num_states), range(self.mf.num_states)):
            ref_phi = self.ref_mf.phi(self.x)
            ref_mltp = self.ref_mf.marginal_ltp(i, j, self.k, ref_phi)

            phi = self.mf.phi(self.x)
            mltp = self.mf.marginal_ltp(i, j, self.k, phi)
            self.assertAlmostEqual(ref_mltp, mltp, places=6)

    def test_phi(self):
        self.scenario_uni1()
        ref_phi = self.x[self.k]
        np.testing.assert_array_equal(ref_phi, self.mf.phi(self.x))
        np.testing.assert_array_equal(ref_phi, self.ref_mf.phi(self.x))

        self.scenario_uni2()
        ref_phi = self.x[self.k]
        np.testing.assert_array_equal(ref_phi, self.mf.phi(self.x))
        np.testing.assert_array_equal(ref_phi, self.ref_mf.phi(self.x))
