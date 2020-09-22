import networkx as nx
import numpy as np

from .config import Config


class NetworkConfig(Config):
    @classmethod
    def erdosrenyi(cls, num_nodes=1000, p=0.004, weights=None, num_layers=None):
        cls = cls()
        cls.name = "ERNetwork"
        cls.num_nodes = num_nodes
        cls.p = p
        if weights is not None:
            cls.weights = weights

        if isinstance(num_layers, int):
            cls.layers = [f"layer{i}" for i in range(num_layers)]

        return cls

    @classmethod
    def barabasialbert(cls, num_nodes=1000, m=2, p=-1, weights=None, num_layers=None):
        cls = cls()
        cls.name = "BANetwork"
        cls.num_nodes = num_nodes
        cls.m = m
        cls.p = p
        if weights is not None:
            cls.weights = weights

        if isinstance(num_layers, int):
            cls.layers = [f"layer{i}" for i in range(num_layers)]
        return cls

    @classmethod
    def configuration(cls, num_nodes, p_k):
        cls = cls()
        cls.name = "ConfigurationNetwork"
        cls.num_nodes = num_nodes
        cls.p_k = p_k
        return cls

    @classmethod
    def spain_mobility(cls, path, weighted=False, mutliplex=False):
        cls = cls()
        cls.name = "RealNetwork"
        cls.path = path
        if weighted and multiplex:
            cls.group_name = "weighted-multiplex"
        elif weighted and not multiplex:
            cls.group_name = "weighted"
        elif not weighted and multiplex:
            cls.group_name = "multiplex"
        else:
            cls.group_name = "thresholded"

        return cls

    @classmethod
    def realnetwork(cls, path_to_edgelist):
        cls = cls()
        cls.name = "RealNetwork"
        cls.edgelist = np.loadtxt(path_to_edgelist, dtype=np.int)
        cls.num_nodes = np.unique(cls.edgelist.flatten()).shape[0]
        return cls

    @classmethod
    def realtemporalnetwork(cls, path_to_edgelist, window=1):
        cls = cls()
        cls.name = "RealTemporalNetwork"
        cls.edges = np.loadtxt(path_to_edgelist).astype("int")
        t = np.unique(cls.edges)
        cls.dt = np.min(np.abs(t - np.roll(t, -1))[:-1])
        cls.window = int(3600 / cls.dt * window)
        cls.num_nodes = np.unique(cls.edges[:, :2].flatten()).shape[0]
        return cls

    @property
    def is_weighted(self):
        return "weights" in self.__dict__

    @property
    def is_multiplex(self):
        return "layers" in self.__dict__


class NetworkWeightConfig(Config):
    @classmethod
    def uniform(cls):
        cls = cls()
        cls.name = "UniformWeightGenerator"
        cls.low = 0
        cls.high = 100
        return cls

    @classmethod
    def loguniform(cls):
        cls = cls()
        cls.name = "LogUniformWeightGenerator"
        cls.low = 1e-5
        cls.high = 100
        return cls

    @classmethod
    def normal(cls):
        cls = cls()
        cls.name = "NormalWeightGenerator"
        cls.mean = 100
        cls.std = 5
        return cls

    @classmethod
    def lognormal(cls):
        cls = cls()
        cls.name = "LogNormalWeightGenerator"
        cls.mean = 100
        cls.std = 5
        return cls

    @classmethod
    def degree(cls):
        cls = cls()
        cls.name = "DegreeWeightGenerator"
        cls.mean = 100
        cls.std = 5
        cls.normalized = True
        return cls

    @classmethod
    def betweenness(cls):
        cls = cls()
        cls.name = "BetweennessWeightGenerator"
        cls.mean = 100
        cls.std = 5
        cls.normalized = True
        return cls
