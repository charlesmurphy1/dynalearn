import networkx as nx
import numpy as np
import torch

from dynalearn.datasets import RemapStateTransform
from unittest import TestCase


class RemapStateTransformTest(TestCase):
    def setUp(self):
        self.transform = RemapStateTransform()
        self.transform.state_map = {0: 0, 1: 1, 2: 0, 3: 1}
        return

    def test_call(self):
        x = np.random.randint(4, size=1000)
        y = self.transform(x)
        x_ref = x * 1
        x_ref[x == 2] = 0
        x_ref[x == 3] = 1
        np.testing.assert_array_equal(x_ref, y)
