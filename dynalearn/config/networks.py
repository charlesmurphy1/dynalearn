import networkx as nx
import numpy as np

from .config import Config


class NetworkConfig(Config):
    @classmethod
    def erdosrenyi(cls, num_nodes, p):
        cls = cls()
        cls.name = "ERNetwork"
        cls.num_nodes = num_nodes
        cls.p = p
        return cls

    @classmethod
    def er_default(cls):
        cls = cls()
        for k, v in NetworkConfig.erdosrenyi(1000, 0.004).__dict__.items():
            cls.__dict__[k] = v
        return cls

    @classmethod
    def barabasialbert(cls, num_nodes, m):
        cls = cls()
        cls.name = "BANetwork"
        cls.num_nodes = num_nodes
        cls.m = m
        return cls

    @classmethod
    def ba_default(cls):
        cls = cls()
        for k, v in NetworkConfig.barabasialbert(1000, 2).__dict__.items():
            cls.__dict__[k] = v
        return cls

    @classmethod
    def treeba_default(cls):
        cls = cls()
        for k, v in NetworkConfig.barabasialbert(1000, 1).__dict__.items():
            cls.__dict__[k] = v
        return cls

    @classmethod
    def configuration(cls, num_nodes, p_k):
        cls = cls()
        cls.name = "ConfigurationNetwork"
        cls.num_nodes = num_nodes
        cls.p_k = p_k
        return cls

    @classmethod
    def realnetwork(cls, path_to_edgelist):
        cls.name = "RealNetwork"
        cls.edgelist = np.loadtxt(path_to_edgelist, dtype=np.int)
        cls.num_nodes = np.unique(cls.edgelist.flatten()).shape[0]
        return cls

    @classmethod
    def realtemporalnetwork(cls, path_to_edgelist, window=1):
        cls.name = "RealTemporalNetwork"
        cls.edges = np.loadtxt(path_to_edgelist).astype("int")
        t = np.unique(cls.edges)
        cls.dt = np.min(np.abs(t - np.roll(t, -1))[:-1])
        cls.window = int(3600 / cls.dt * window)
        cls.num_nodes = np.unique(cls.edges[:, :2].flatten()).shape[0]
        return cls
