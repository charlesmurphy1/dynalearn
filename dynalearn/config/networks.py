import networkx as nx
import numpy as np

from dynalearn.utilities import Config


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
        return NetworkConfig.erdosrenyi(1000, 0.004)

    @classmethod
    def barabasialbert(cls, num_nodes, m):
        cls.name = "BANetwork"
        cls.num_nodes = num_nodes
        cls.m = m
        return cls

    @classmethod
    def ba_default(cls):
        return NetworkConfig.barabasialbert(1000, 2)

    @classmethod
    def configuration(cls, num_nodes, p_k):
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
    def realtemporalnetwork(cls, path_to_edgelist, window, dt):
        cls.name = "RealTemporalNetwork"
        cls.edges = np.loadtxt(path_to_edgelist, dtype=np.int)
        cls.window = window
        cls.dt = dt
        cls.num_nodes = np.unique(cls.edges[:, :2].flatten()).shape[0]
        return cls
