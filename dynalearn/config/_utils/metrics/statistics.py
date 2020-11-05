import numpy as np

from dynalearn.config import Config


class StatisticsConfig(Config):
    @classmethod
    def default(cls):
        cls = cls()
        cls.max_num_points = 10000
        cls.max_window_size = 3
        return cls
