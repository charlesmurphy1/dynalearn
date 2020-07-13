import numpy as np

from unittest import TestCase
from dynalearn.datasets import Data, WindowedData


class TestData(TestCase):
    def setUp(self):
        self.name = "data0"
        self.x = np.random.rand(10, 5, 6)
        self.window_size = 2
        self.window_step = 2
        self.data = Data(name=self.name, data=self.x)
        self.w_data = WindowedData(
            name=self.name,
            data=self.x,
            window_size=self.window_size,
            window_step=self.window_step,
        )

    def test_call(self):
        index = 4
        back = (self.window_size - 1) * self.window_step
        np.testing.assert_array_equal(self.data[index], self.x[index])
        np.testing.assert_array_equal(
            self.w_data[index], self.x[index - back : index + 1 : self.window_step]
        )

    def test_copy(self):
        data_copy = self.data.copy()
        self.assertEqual(data_copy.__dict__, self.data.__dict__)
        data_copy = self.w_data.copy()
        self.assertEqual(data_copy.__dict__, self.w_data.__dict__)

    def test_add(self):
        data = Data(name="new_data")
        data.add(self.x)
        np.testing.assert_array_equal(data.data, self.data.data)


if __name__ == "__main__":
    unittest.main()
