import networkx as nx
import numpy as np

from abc import abstractmethod
from dynalearn.config import Config
from dynalearn.networks.network import Network
from dynalearn.networks.transform import NetworkTransformList


class GenerativeNetwork(Network):
    def __init__(self, config=None, weight_gen=None, transforms=[], **kwargs):
        config = config or Config(**kwargs)
        Network.__init__(self, config)
        self.weight_gen = weight_gen
        self.transforms = NetworkTransformList(transforms)
        if self.weight_gen is not None:
            self.is_weighted = True

    def generate(self, seed=None):
        if seed is None:
            seed = np.random.randint(2 ** 31)
        if self.layers is not None:
            g = {}
            for l in self.layers:
                g[l] = self.net_gen(seed)
                if self.weight_gen is not None:
                    g[l] = self.weight_gen(g[l])
                g[l] = self.transforms(g[l])

        else:
            g = self.net_gen(seed)
            if self.weight_gen is not None:
                g = self.weight_gen(g)
            g = self.transforms(g)
        return g


class GNPNetwork(GenerativeNetwork):
    def net_gen(self, seed=None):
        return nx.gnp_random_graph(self.num_nodes, self.config.p, seed=seed)


class GNMNetwork(GenerativeNetwork):
    def net_gen(self, seed=None):
        return nx.gnm_random_graph(self.num_nodes, self.config.m, seed=seed)


class BANetwork(GenerativeNetwork):
    def net_gen(self, seed=None):
        return nx.barabasi_albert_graph(self.num_nodes, self.config.m, seed)


class ConfigurationNetwork(GenerativeNetwork):
    def __init__(self, config=None, weight_gen=None, **kwargs):
        config = config or Config(**kwargs)
        GenerativeNetwork.__init__(self, config, weight_gen=weight_gen, **kwargs)
        self.p_k = config.p_k
        if "maxiter" in config.__dict__:
            self.maxiter = config.maxiter
        else:
            self.maxiter = 100

    def net_gen(self, seed=None):
        if "maxiter" in self.config.__dict__:
            maxiter = self.config.maxiter
        else:
            maxiter = 100
        it = 0
        while it < maxiter:
            seq = self.p_k.sample(self.num_nodes)
            if np.sum(seq) % 2 == 0:
                g = nx.expected_degree_graph(seq, seed=seed)
                return g
            it += 1
        raise ValueError("Invalid degree sequence.")
