import networkx as nx
import numpy as np

from abc import abstractmethod
from dynalearn.networks.base import Network


class GenerativeNetwork(Network):
    @abstractmethod
    def generate(self, seed=None):
        raise NotImplementedError()


class ERNetwork(GenerativeNetwork):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        GenerativeNetwork.__init__(self, config)
        self.p = config.p

    def generate(self, seed=None):
        if seed is None:
            seed = np.random.randint(2 ** 31)
        g = nx.gnp_random_graph(self.num_nodes, self.p)
        self.data.append(g)
        return g


class BANetwork(GenerativeNetwork):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        GenerativeNetwork.__init__(self, config)
        self.m = config.m

    def generate(self, seed=None):
        if seed is None:
            seed = np.random.randint(2 ** 31)
        g = nx.barabasi_albert_graph(self.num_nodes, self.m)
        self.data.append(g)
        return g


class ConfigurationNetwork(GenerativeNetwork):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        GenerativeNetwork.__init__(self, config)
        self.p_k = config.p_k
        if "maxiter" in config.__dict__:
            self.maxiter = config.maxiter
        else:
            self.maxiter = 100

    def generate(self, seed=None):
        if seed is None:
            seed = np.random.randint(2 ** 31)
        it = 0
        while it < self.maxiter:
            seq = self.p_k.sample(self.num_nodes)
            if np.sum(seq) % 2 == 0:
                g = nx.configuration_model(seq, seed=None)
                self.data.append(g)
                return g
            it += 1
        raise ValueError("Invalid degree sequence.")
