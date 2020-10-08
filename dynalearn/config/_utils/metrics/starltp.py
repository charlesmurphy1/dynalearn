import numpy as np

from dynalearn.config import Config


class StarLTPConfig(Config):
    @classmethod
    def test(cls):
        cls = cls()
        cls.degree = np.linspace(1, 20, 5).astype("int")
        return cls

    @classmethod
    def default(cls):
        cls = cls()
        cls.degree = np.unique(np.logspace(0, 2, 30).astype("int"))
        return cls
