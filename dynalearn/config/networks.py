import networkx as nx
import numpy as np

from .config import Config


class NetworkConfig(Config):
    @classmethod
    def gnp(
        cls, num_nodes=1000, p=0.004, weights=None, transforms=None, num_layers=None
    ):
        cls = cls()
        cls.name = "GNPNetwork"
        cls.num_nodes = num_nodes
        cls.p = p
        if weights is not None:
            cls.weights = weights
        if transforms is not None:
            cls.transforms = transforms

        if isinstance(num_layers, int):
            cls.layers = [f"layer{i}" for i in range(num_layers)]

        return cls

    @classmethod
    def gnm(
        cls, num_nodes=1000, m=2000, weights=None, transforms=None, num_layers=None
    ):
        cls = cls()
        cls.name = "GNMNetwork"
        cls.num_nodes = num_nodes
        cls.m = m
        if weights is not None:
            cls.weights = weights
        if transforms is not None:
            cls.transforms = transforms

        if isinstance(num_layers, int):
            cls.layers = [f"layer{i}" for i in range(num_layers)]

        return cls

    @classmethod
    def barabasialbert(
        cls, num_nodes=1000, m=2, weights=None, transforms=None, num_layers=None
    ):
        cls = cls()
        cls.name = "BANetwork"
        cls.num_nodes = num_nodes
        cls.m = m
        if weights is not None:
            cls.weights = weights
        if transforms is not None:
            cls.transforms = transforms

        if isinstance(num_layers, int):
            cls.layers = [f"layer{i}" for i in range(num_layers)]
        return cls

    @classmethod
    def w_gnp(cls, num_nodes=1000, p=0.004):
        w = NetworkWeightConfig.uniform()
        t = NetworkTransformConfig.sparcifier()
        cls = cls.gnp(num_nodes=num_nodes, p=p, weights=w, transforms=t)
        return cls

    @classmethod
    def w_ba(cls, num_nodes=1000, m=2):
        w = NetworkWeightConfig.uniform()
        t = NetworkTransformConfig.sparcifier()
        cls = cls.barabasialbert(num_nodes=num_nodes, m=m, weights=w, transforms=t)
        return cls

    @classmethod
    def mw_ba(cls, num_nodes=1000, m=2, num_layers=1):
        w = NetworkWeightConfig.uniform()
        t = NetworkTransformConfig.sparcifier()
        cls = cls.barabasialbert(
            num_nodes=num_nodes, m=m, weights=w, transforms=t, num_layers=num_layers
        )
        return cls

    @property
    def is_weighted(self):
        return "weights" in self.__dict__

    @property
    def is_multiplex(self):
        return "layers" in self.__dict__


class NetworkTransformConfig(Config):
    @classmethod
    def sparcifier(cls):
        cls = cls()
        cls.names = ["SparcifierTransform"]
        cls.maxiter = 100
        cls.p = -1
        return cls


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
