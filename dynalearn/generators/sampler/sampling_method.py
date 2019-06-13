import numpy as np

import numpy as np


class DiscreteSampling(object):
    """docstring for DiscreteSampling."""

    def __init__(self, data_dict=None):
        super(DiscreteSampling, self).__init__()
        self._values = None
        self._weights = None
        if data_dict is not None:
            self.set_data(data_dict)

    def sample(self, size=None):

        if size is None:
            size = 1

        r = np.random.rand(size)
        index = np.searchsorted(self._cumulative, r)
        val = self.values[index]
        if size == 1:
            return val[0]
        else:
            return val

    @property
    def weights(self):
        if self._weights is None:
            raise ValueError("No weights available.")
        else:
            return self._weights

    @weights.setter
    def weights(self, weights):
        raise ValueError("Use self.set_data() to change values and weights jointly.")

    @property
    def values(self):
        if self._values is None:
            raise ValueError("No values available.")
        else:
            return self._values

    @values.setter
    def values(self, values):
        raise ValueError("Use self.set_data() to change values and weights.")

    def set_data(self, data_dict):
        self._values = np.array(list(data_dict.keys()))
        self._weights = np.array(list(data_dict.values()))
        self._cumulative = np.cumsum(self._weights) / np.sum(self._weights)


class BernoulliSampling(object):
    """docstring for BernoulliSampling."""

    def __init__(self, weights=None):
        super(BernoulliSampling, self).__init__()

        self._weights = None
        if weights is not None:
            self.weights = weights

    def sample(self, size=None):
        if size is None:
            size = 1
        r = np.random.rand(size, *self.weights.shape)
        val = (r < self.weights).astype("float")

        if size == 1:
            return val.reshape(*self.weights.shape)
        else:
            return val

    @property
    def weights(self):
        if self._weights is None:
            raise ValueError("No weights available.")
        else:
            return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = np.array(weights)
