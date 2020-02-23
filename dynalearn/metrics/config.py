import numpy as np
from dynalearn.metrics.aggregator import *


class MetricsConfig:
    @classmethod
    def SISMetrics(cls):

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
        cls.num_k = 7
        cls.mf_epsilon = 1e-2
        cls.tol = 1e-4
        cls.p_range = (0.1, 10)
        cls.fp_finder = None
        cls.discontinuous = False

        # stationary states metrics
        cls.num_nodes = 10000
        cls.ss_epsilon = 10.0 / cls.num_nodes
        cls.ss_parameters = np.concatenate(
            (np.linspace(0.1, 3, 30), np.linspace(3.1, 10, 20))
        )
        cls.num_samples = 25
        cls.initial_burn = 200
        cls.burn = 10
        cls.reshuffle = 10
        cls.fp_finder = None

        return cls

    @classmethod
    def PlanckSISMetrics(cls):

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
            (np.linspace(0.1, 2, 10), np.linspace(2.1, 6, 50), np.linspace(6.1, 10, 10))
        )
        cls.num_k = 7
        cls.mf_epsilon = 1e-2
        cls.tol = 1e-4
        cls.p_range = (0.1, 10)
        cls.fp_finder = None
        cls.discontinuous = True

        # stationary states metrics
        cls.num_nodes = 10000
        cls.ss_epsilon = 10.0 / cls.num_nodes
        cls.ss_parameters = np.concatenate(
            (np.linspace(0.1, 2, 10), np.linspace(2.1, 6, 50), np.linspace(6.1, 10, 10))
        )
        cls.num_samples = 25
        cls.initial_burn = 200
        cls.burn = 10
        cls.reshuffle = 10
        cls.fp_finder = None

        return cls

    @classmethod
    def SISSISMetrics(cls):

        cls = cls()

        # ltp, attention and statistics metrics
        cls.num_points = 1000
        cls.mle_num_points = None
        cls.att_num_points = 100
        cls.max_num_sample = 1000
        cls.aggregator = InteractingContagionAggregator()
        cls.degree_class = np.unique(np.logspace(0, 2, 30).astype("int"))

        # meanfield metrics
        cls.mf_parameters = np.concatenate(
            (np.linspace(0.1, 5, 40), np.linspace(6.1, 10, 10))
        )
        cls.num_k = 7
        cls.mf_epsilon = 1e-2
        cls.tol = 1e-4
        cls.p_range = (0.1, 10)
        cls.fp_finder = None
        cls.discontinuous = True

        # stationary states metrics
        cls.num_nodes = 10000
        cls.ss_epsilon = 10.0 / cls.num_nodes
        cls.ss_parameters = np.concatenate(
            (np.linspace(0.1, 5, 40), np.linspace(6.1, 10, 10))
        )
        cls.num_samples = 25
        cls.initial_burn = 200
        cls.burn = 10
        cls.reshuffle = 10
        cls.fp_finder = None

        return cls
