import h5py
import numpy as np
from abc import ABC, abstractmethod

from dynalearn.datasets.data.data import Data


class StateData(Data):
    def __init__(self, name="state_data", data=None):
        Data.__init__(self, name=name)
        if data is not None:
            self.data = data

    def __eq__(self, other):
        if isinstance(other, StateData):
            return np.all(self.data == other.data)
        return False

    def get(self, index):
        return self._data[index]

    @property
    def size(self):
        return self._data.shape[0]

    @property
    def shape(self):
        return self._data.shape[1:]
