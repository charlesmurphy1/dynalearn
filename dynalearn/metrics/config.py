import numpy as np
from dynalearn.metrics.aggregator import *


class MetricsConfig:
    @classmethod
    def default(cls):

        cls = cls()

        # ltp, attention and statistics metrics
        cls.num_points = 1000
        cls.mle_num_points = None
        cls.att_num_points = 100
        cls.max_num_sample = 1000
        cls.aggregator = SimpleContagionAggregator()
        cls.degree_class = np.unique(np.logspace(0, 2, 30).astype("int"))

        # meanfield metrics
        cls.mf_parameters = np.concatenate(
            (np.linspace(0.1, 3, 50), np.linspace(3.1, 10, 20))
        )
        cls.num_k = 5
        cls.epsilon = 1e-2
        cls.tol = 1e-3
        cls.p_range = (0.1, 3)
        cls.fp_finder = None

        # stationary states metrics
        cls.num_nodes = 2000
        cls.ss_parameters = np.linspace(0.1, 10, 10)
        cls.num_samples = 100
        cls.burn = 100
        cls.reshuffle = 0.1
        cls.tol = 1e-3

        return cls

    @classmethod
    def SISSISMetrics(cls):

        cls = cls()

        # ltp, attention and statistics metrics
        cls.num_points = 1000
        cls.mle_num_points = None
        cls.att_num_points = 100
        cls.max_num_sample = 1000
        cls.aggregator = InteractingContagionAggregatoreContagionAggregator()
        cls.degree_class = np.unique(np.logspace(0, 2, 30).astype("int"))

        # meanfield metrics
        cls.mf_parameters = np.concatenate(
            np.linspace(0.1, 3, 50), np.linspace(3.1, 10, 20)
        )
        cls.num_k = 5
        cls.epsilon = 1e-2
        cls.tol = 1e-3
        cls.p_range = (0.1, 3)
        cls.fp_finder = None

        # stationary states metrics
        cls.num_nodes = 2000
        cls.ss_parameters = np.linspace(0.1, 10, 10)
        cls.num_samples = 100
        cls.burn = 100
        cls.reshuffle = 0.1
        cls.tol = 1e-3

        return cls

    @classmethod
    def test(cls):

        cls = cls()

        # ltp, attention and statistics metrics
        cls.num_points = 1000
        cls.mle_num_points = 1000
        cls.att_num_points = 100
        cls.max_num_sample = 1000
        cls.aggregator = SimpleContagionAggregator()
        cls.degree_class = np.unique(np.logspace(0, 2, 30).astype("int"))

        # meanfield metrics
        cls.mf_parameters = np.concatenate(
            (np.linspace(0.1, 3, 30), np.linspace(3.1, 10, 10))
        )
        cls.num_k = 5
        cls.epsilon = 1e-2
        cls.tol = 1e-3
        cls.p_range = (0.1, 3)
        cls.fp_finder = None

        # stationary states metrics
        cls.num_nodes = 1000
        cls.ss_parameters = np.linspace(0.1, 10, 10)
        cls.num_samples = 10
        cls.burn = 100
        cls.reshuffle = 0.1
        cls.tol = 1e-3

        return cls
