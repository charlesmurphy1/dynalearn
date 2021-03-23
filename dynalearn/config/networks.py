import networkx as nx
import numpy as np

from .config import Config


class NetworkConfig(Config):
    @classmethod
    def gnp(cls, num_nodes=1000, p=0.004, weights=None, transforms=None, layers=None):
        cls = cls()
        cls.name = "GNPNetworkGenerator"
        cls.num_nodes = num_nodes
        cls.p = p
        if weights is not None:
            cls.weights = weights
        if transforms is not None:
            cls.transforms = transforms

        if isinstance(layers, int):
            cls.layers = [f"layer{i}" for i in range(layers)]
        elif isinstance(layers, list):
            cls.layers = layers

        return cls

    @classmethod
    def gnm(cls, num_nodes=1000, m=2000, weights=None, transforms=None, layers=None):
        cls = cls()
        cls.name = "GNMNetworkGenerator"
        cls.num_nodes = num_nodes
        cls.m = m
        if weights is not None:
            cls.weights = weights
        if transforms is not None:
            cls.transforms = transforms

        if isinstance(layers, int):
            cls.layers = [f"layer{i}" for i in range(layers)]
        elif isinstance(layers, list):
            cls.layers = layers

        return cls

    @classmethod
    def barabasialbert(
        cls, num_nodes=1000, m=2, weights=None, transforms=None, layers=None
    ):
        cls = cls()
        cls.name = "BANetworkGenerator"
        cls.num_nodes = num_nodes
        cls.m = m
        if weights is not None:
            cls.weights = weights
        if transforms is not None:
            cls.transforms = transforms
        if isinstance(layers, int):
            cls.layers = [f"layer{i}" for i in range(layers)]
        elif isinstance(layers, list):
            cls.layers = layers

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
    def mw_ba(cls, num_nodes=1000, m=2, layers=1):
        w = NetworkWeightConfig.uniform()
        t = NetworkTransformConfig.sparcifier()
        cls = cls.barabasialbert(
            num_nodes=num_nodes, m=m, weights=w, transforms=t, layers=layers
        )
        return cls

    @classmethod
    def covid_pretrain(cls, num_nodes=1000, m=2):
        w = NetworkWeightConfig.uniform()
        t = NetworkTransformConfig.sparcifier()
        l = ["plane", "car", "bus", "boat", "train"]
        cls = cls.barabasialbert(
            num_nodes=num_nodes, m=m, weights=w, transforms=t, layers=l
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
