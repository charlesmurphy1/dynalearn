import unittest

import sys
import torch
sys.path.append("../")
from param import *
from unit import *



class TestWeightClass(unittest.TestCase):
    """docstring for TestWeightClass"""
    def __init__(self, *args, **kwargs):
        batchsize = 2
        use_cuda = False

        key = "v"
        size = 3
        kind = "visible"
        self.u1 = Unit(key, Unit_info(key, size, kind),batchsize, use_cuda)

        key = "h"
        size = 4
        kind = "hidden"
        self.u2 = Unit(key, Unit_info(key, size, kind),batchsize, use_cuda)

        self.weight = Weight((self.u1, self.u2), batchsize, init_mode=None,
                             use_cuda=use_cuda)


        super(TestWeightClass, self).__init__(*args, **kwargs)


    def test_repr(self):
        self.assertEqual(self.weight.__repr__(), "<param.Param.Weight>")


    def test_init_none(self):
        self.weight.init_value()
        self.assertTrue(torch.all(self.weight.value == torch.zeros(3, 4)))


if __name__ == '__main__':
    unittest.main()