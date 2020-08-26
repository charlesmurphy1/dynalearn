import h5py
import numpy as np
import torch

from abc import ABC, abstractmethod
from dynalearn.datasets.data.data import Data
from dynalearn.utilities import to_edge_index, get_edge_attr, set_edge_attr, onehot


class NetworkData(Data):
    def __init__(self, name="network_data", data=None):
        Data.__init__(self, name=name)
        if data is not None:
            self.data = data
        else:
            self._data = None
            self.num_nodes = None

    def __eq__(self, other):
        if isinstance(other, NetworkData):
            if isinstance(self.data, dict):
                for k, v in self.data.items():
                    if k not in other.data:
                        return False
                    elif v.edges() != other.data[k].edges():
                        return False
                return True
            else:
                return self.data.edges() == other.data.edges()
        else:
            return False

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
        self.num_nodes = None
        if isinstance(self._data, dict):
            for k, g in self._data.items():
                if self.num_nodes is None:
                    self.num_nodes = g.number_of_nodes()
                else:
                    assert (
                        self.num_nodes == g.number_of_nodes()
                    ), "All networks must have the same node set."

    def get(self):
        return self.data

    def save(self, h5file):
        group = h5file.create_group(self.name)
        if isinstance(self.data, dict):
            for k, v in self.data.items():
                _group = group.create_group(k)
                self._save_graph_(v, _group)
        else:
            self._save_graph_(self.data, group)

    def load(self, h5file):
        if self.name not in h5file:
            return
        group = h5file[self.name]
        if "edge_list" in group:
            self.data = self._load_graph_(group)
        else:
            self.data = {}
            for k in group.keys():
                if "edge_list" not in group:
                    raise ValueError(f"No edge list found while loading {self.name}.")
                self.data[k] = self._load_graph_(group[k])

    def _save_graph_(self, g, h5file):
        if len(g.edges()) > 0:
            edge_list = to_edge_index(g).numpy().T
            edge_attr = get_edge_attr(g)
        else:
            edge_list = np.zeros((0, 2)).astype("int")
            edge_attr = {}
        h5file.create_dataset("edge_list", data=edge_list)
        for k, v in edge_attr.items():
            h5file.create_dataset(k, data=v)

    def _load_graph_(self, h5file):
        edge_list = h5file["edge_list"][...]
        g = nx.from_edgelist(edge_list)
        edge_attr = {}
        for k, v in h5file.items():
            if k != "edge_list":
                edge_attr[k] = v
        g = set_edge_attr(g, edge_attr)
        return g
