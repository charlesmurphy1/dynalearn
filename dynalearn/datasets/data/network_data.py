import h5py
import networkx as nx
import numpy as np
import torch

from abc import ABC, abstractmethod
from dynalearn.datasets.data.data import Data
from dynalearn.utilities import (
    to_edge_index,
    get_edge_attr,
    set_edge_attr,
    get_node_attr,
    set_node_attr,
    onehot,
)


class NetworkData(Data):
    def __init__(self, name="network_data", data=None):
        Data.__init__(self, name=name)
        if data is not None:
            if isinstance(data, h5py.Group):
                data = self._load_graph_(data)
            assert issubclass(data.__class__, nx.Graph)
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
        if self.name in h5file:
            group = h5file[self.name]
        else:
            group = h5file
        if "edge_list" in group:
            self.data = self._load_graph_(group)
        else:
            self.data = {}
            for k in group.keys():
                if "edge_list" not in group:
                    raise ValueError(f"No edge list found while loading {self.name}.")
                self.data[k] = self._load_graph_(group[k])

    def _save_graph_(self, g, h5file):
        node_list = np.array(g.nodes())
        node_attr = get_node_attr(g)
        h5file.create_dataset("node_list", data=node_list)
        node_group = h5file.create_group("node_attr")
        for k, v in node_attr.items():
            node_group.create_dataset(k, data=v)

        if len(g.edges()) > 0:
            edge_list = to_edge_index(g).T
            edge_attr = get_edge_attr(g)
        else:
            edge_list = np.zeros((0, 2)).astype("int")
            edge_attr = {}
        h5file.create_dataset("edge_list", data=edge_list)
        edge_group = h5file.create_group("edge_attr")
        for k, v in edge_attr.items():
            edge_group.create_dataset(k, data=v)

    def _load_graph_(self, h5file):
        node_list = h5file["node_list"][...]
        edge_list = h5file["edge_list"][...]

        g = nx.DiGraph()
        g.add_nodes_from(node_list)
        g.add_edges_from(edge_list)
        edge_attr = {}
        if "edge_attr" in h5file:
            for k, v in h5file["edge_attr"].items():
                edge_attr[k] = v
            g = set_edge_attr(g, edge_attr)

        node_attr = {}
        if "node_attr" in h5file:
            for k, v in h5file["node_attr"].items():
                node_attr[k] = v
            g = set_node_attr(g, node_attr)
        return g
