import numpy as np
from dynalearn.utilities import to_edge_index, onehot


class Data:
    def __init__(self, name="data", data=None, shape=None):
        self.name = name
        self.data = data
        if shape is not None:
            self.shape = shape
        elif data is not None:
            self.shape = data.shape[1:]
        else:
            self.shape = ()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.size * np.prod(self.shape)

    def add(self, x):
        if self.data is None:
            self.data = np.zeros((0, *self.shape))
        if self.shape == x.shape:
            x = x.reshape(1, *x.shape)
            self.data = np.concatenate((self.data, x), axis=0)
        elif self.shape == x.shape[1:]:
            self.data = np.concatenate((self.data, x), axis=0)
        else:
            raise ValueError(f"Shape {x.shape} is invalid, expected shape {self.shape}")

    def copy(self):
        data_copy = self.__class__()
        data_copy.__dict__ = self.__dict__.copy()
        data_copy.data = self.data.copy()
        return data_copy

    def transform(self, t):
        for i, x in enumerate(self.data):
            self.data[i] = t(x)

    @property
    def size(self):
        if self.data is not None:
            return self.data.shape[0]
        else:
            return 0


class WindowedData(Data):
    def __init__(
        self, name="data", data=None, window_size=1, window_step=1, shape=None
    ):
        self.window_size = window_size
        self.window_step = window_step
        self.back = (self.window_size - 1) * self.window_step
        Data.__init__(self, name=name, data=data, shape=None)

    def __getitem__(self, index):
        if index < self.back:
            raise ValueError(
                f"Invalid index {index}. For windowed data, index must be (>=) to {self.back}."
            )
        index += self.delay
        x = self.data[
            index
            - self.window_size * self.window_step : index
            - self.window_step
            + 1 : self.window_step
        ]
        return x


class NetworkData(Data):
    def __init__(self, name="data", data=None):
        Data.__init__(self, name="data", data=data)
        if self.data is not None:
            self.edge_index_list = [to_edge_index(g) for g in self.data]
        else:
            self.edge_index_list = []
            self.data = []

    def __getitem__(self, index):
        return self.edge_index_list[index]

    def add(self, g):
        self.data.append(g)
        self.edge_index_list.append(to_edge_index(g))

    def transform(self, tranformation):
        for i, g in enumerate(self.data):
            self.data[i] = tranformation(g)
        self.edge_index_list = [to_edge_index(g) for g in self.data]

    @property
    def size(self):
        if self.data is not None:
            return len(self.data)
        else:
            return 0
