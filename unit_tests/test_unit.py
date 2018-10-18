import unittest

import sys
import torch
sys.path.append("../")
from unit import *


class TestUnitInfoClass(unittest.TestCase):
    """docstring for TestUnitClass"""
    def __init__(self, *args, **kwargs):
        key = "visible_unit"
        size = 3
        kind = "visible"

        self.unit_info = Unit_info(key, size, kind)

        super(TestUnitInfoClass, self).__init__(*args, **kwargs)


    def test_repr(self):
        self.assertEqual(self.unit_info.__repr__(), "<unit.Unit_info>")


    def test_str(self):
        self.assertEqual(self.unit_info.__str__(), "visible_unit 3 visible")


class TestUnitClass(unittest.TestCase):
    """docstring for TestUnitClass"""
    def __init__(self, *args, **kwargs):
        key = "visible_unit"
        size = 3
        kind = "visible"
        batchsize = 2
        use_cuda = False


        self.unit_info = Unit_info(key, size, kind)
        self.unit = Unit(key, self.unit_info, batchsize, use_cuda)

        value = [[1, 2, 3],[4, 5, 6]]
        self.value = torch.Tensor(value)
        self.unit.value = self.value

        super(TestUnitClass, self).__init__(*args, **kwargs)


    def test_repr(self):
        self.assertEqual(self.unit.__repr__(), "<unit.Unit.single>")
        

    def test_str(self):
        self.assertEqual(self.unit.__str__(), str(self.value))


if __name__ == '__main__':
    unittest.main()
        
