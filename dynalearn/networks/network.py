import numpy as np
import networkx as nx

from abc import ABC, abstractmethod
from dynalearn.config import Config


class Network(ABC):
    def __init__(self, config=None):
        if config is None:
            config = Config()
        self._config = config
        self.data = []
        if "num_nodes" in config.__dict__:
            self.num_nodes = config.num_nodes
        else:
            self.num_nodes = None
        if "layers" in config.__dict__:
            self.layers = config.layers
        else:
            self.layers = None
        self.is_weighted = False

    @property
    def config(self):
        return self._config

    def clear(self):
        self.data = []
