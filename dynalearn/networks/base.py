import numpy as np
import networkx as nx

from abc import ABC, abstractmethod


class Network(ABC):
    def __init__(self, config):
        self._config = config
        self._num_nodes = config.num_nodes
        self.data = []

    @property
    def num_nodes(self):
        return self._num_nodes

    def clear(self):
        self.data = []
