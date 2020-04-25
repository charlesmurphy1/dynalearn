import numpy as np
import unittest

from dynalearn.meanfields import RecurrenceFPF, NewtonFPF, HybridRecurrenceNewtonFPF


class RecurrenceTest(unittest.TestCase):
    def setUp(self):
        self.tol = 1e-6
        self.iteration = 10
        self.max_iter = 1000
        self.function = lambda x: x / np.pi + 1
        self.finder1 = RecurrenceFPF(
            tol=self.tol, initial_iter=self.iteration, max_iter=self.max_iter
        )
        self.finder2 = NewtonFPF(tol=self.tol, max_iter=self.max_iter)
        self.finder3 = HybridRecurrenceNewtonFPF(
            tol=self.tol, rec_iter=self.iteration, max_iter=self.max_iter
        )

    def test_recurrence(self):
        x0 = np.random.randn(5)
        result = self.finder1(self.function, x0)
        ref_x = np.ones(5) * np.pi / (np.pi - 1)
        x = result.x
        self.assertTrue(result.success)
        np.testing.assert_array_almost_equal(ref_x, x)

    def test_newton(self):
        x0 = np.random.randn(5)
        result = self.finder2(self.function, x0)
        ref_x = np.ones(5) * np.pi / (np.pi - 1)
        x = result.x
        self.assertTrue(result.success)
        np.testing.assert_array_almost_equal(ref_x, x)

    def test_hybrid(self):
        x0 = np.random.randn(5)
        result = self.finder3(self.function, x0)
        ref_x = np.ones(5) * np.pi / (np.pi - 1)
        x = result.x
        self.assertTrue(result.success)
        np.testing.assert_array_almost_equal(ref_x, x)
