import numpy as np
from itertools import product
from .config import Config
from ._utils import LTPConfig, PredictionConfig, StarLTPConfig, StationaryConfig


class MetricsConfig(Config):
    @classmethod
    def test(cls):
        cls = cls()
        cls.names = []
        cls.merge(LTPConfig.default())
        cls.merge(PredictionConfig.default())
        cls.merge(StarLTPConfig.default())
        cls.merge(StationaryConfig.test())

        return cls

    @classmethod
    def sis(cls):
        cls = cls()
        cls.names = []
        cls.merge(LTPConfig.default())
        cls.merge(PredictionConfig.default())
        cls.merge(StarLTPConfig.default())
        cls.merge(StationaryConfig.sis())

        return cls

    @classmethod
    def plancksis(cls):
        cls = cls()
        cls.names = []
        cls.merge(LTPConfig.default())
        cls.merge(PredictionConfig.default())
        cls.merge(StarLTPConfig.default())
        cls.merge(StationaryConfig.plancksis())

        return cls

    @classmethod
    def sissis(cls):
        cls = cls()
        cls.names = []
        cls.merge(LTPConfig.default())
        cls.merge(PredictionConfig.default())
        cls.merge(StarLTPConfig.default())
        cls.merge(StationaryConfig.sissis())

        return cls

    @classmethod
    def dsir(cls):
        cls = cls()
        cls.names = []
        cls.merge(LTPConfig.default())
        cls.merge(PredictionConfig.default())
        cls.merge(StarLTPConfig.default())
        cls.merge(StationaryConfig.dsir())

        return cls
