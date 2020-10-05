import numpy as np
from .config import Config
from itertools import product


class MetricsConfig(Config):
    @classmethod
    def test(cls):
        cls = cls()
        cls.names = []

        # ltp and statistcs metrics
        cls.max_num_points = 1000
        cls.max_num_sample = 1000
        cls.degree_class = np.unique(np.logspace(0, 2, 10).astype("int"))

        # stationary and meanfield metrics
        cls.num_windows = 10
        cls.sample_graph = 0.0
        cls.num_samples = 25
        cls.burn = 10
        cls.adaptive = True
        cls.num_nodes = 1000
        cls.epsilon = 2e-3
        cls.full_data = False
        cls.parameters = np.linspace(0.01, 10.0, 10)

        cls.finder = "RecurrenceFPF"
        cls.num_k = 5
        cls.tol = 1e-6
        cls.max_iter = 5000
        cls.rec_iter = 100
        cls.initial_iter = 100
        cls.with_numba = True

        # summaries
        cls.axis = 1
        cls.err_reduce = "percentile"
        cls.transitions = [(0, 1), (1, 0)]

        return cls

    @classmethod
    def sis(cls):
        cls = cls()
        cls.names = []

        # ltp and statistcs metrics
        cls.max_num_sample = 1000
        cls.max_num_points = 1e4
        # cls.max_num_points = 1000
        cls.degree_class = np.unique(np.logspace(0, 2, 30).astype("int"))

        # stationary and meanfield metrics
        cls.num_samples = 20
        cls.burn = 1000
        cls.adaptive = False
        cls.num_nodes = 2000
        cls.epsilon = 1e-3
        cls.full_data = False
        cls.parameters = np.concatenate(
            (
                np.linspace(0.1, 1.4, 5),
                np.linspace(1.5, 3.5, 50),
                np.linspace(3.6, 7, 15),
            )
        )

        cls.finder = "RecurrenceFPF"
        cls.num_k = 7
        cls.tol = 1e-6
        cls.max_iter = 5000
        cls.rec_iter = 100
        cls.initial_iter = 100
        cls.with_numba = True

        # summaries
        cls.axis = 1
        cls.err_reduce = "percentile"
        cls.transitions = [(0, 1), (1, 0)]

        return cls

    @classmethod
    def plancksis(cls):
        cls = cls()
        cls.names = []

        # ltp and statistcs metrics
        cls.max_num_sample = 1000
        cls.max_num_points = 1e4
        cls.degree_class = np.unique(np.logspace(0, 2, 30).astype("int"))

        # stationary and meanfield metrics
        cls.num_samples = 20
        cls.burn = 1000
        cls.adaptive = False
        cls.num_nodes = 2000
        cls.epsilon = 2e-3
        cls.full_data = False
        cls.parameters = np.concatenate(
            (
                np.linspace(0.1, 3.0, 10),
                np.linspace(3.1, 4.5, 50),
                np.linspace(4.6, 6, 10),
            )
        )

        cls.finder = "RecurrenceFPF"
        cls.num_k = 7
        cls.tol = 1e-6
        cls.max_iter = 5000
        cls.rec_iter = 100
        cls.initial_iter = 100
        cls.with_numba = True

        # summaries
        cls.axis = 1
        cls.err_reduce = "percentile"
        cls.transitions = [(0, 1), (1, 0)]

        return cls

    @classmethod
    def sissis(cls):
        cls = cls()
        cls.names = []

        # ltp and statistcs metrics
        cls.max_num_sample = 1000
        cls.max_num_points = 1e4
        cls.degree_class = np.unique(np.logspace(0, 2, 30).astype("int"))

        # stationary and meanfield metrics
        cls.num_windows = 100
        cls.sample_graph = 0.05
        cls.num_samples = 20
        cls.burn = 1000
        cls.adaptive = True
        cls.num_nodes = 2000
        cls.epsilon = 2e-3
        cls.full_data = False
        cls.parameters = np.concatenate(
            (
                np.linspace(0.1, 1.0, 5),
                np.linspace(1.0, 4.5, 50),
                np.linspace(4.6, 7, 10),
            )
        )

        cls.finder = "RecurrenceFPF"
        cls.num_k = 5
        cls.tol = 1e-6
        cls.max_iter = 5000
        cls.rec_iter = 100
        cls.initial_iter = 100
        cls.with_numba = True

        # summaries
        cls.axis = [1, 3]
        cls.err_reduce = "percentile"
        cls.transitions = [(i, j) for i, j in product(range(4), range(4))]

        return cls

    @classmethod
    def hidden_sissis(cls):
        cls = cls()
        cls.names = []

        # ltp and statistcs metrics
        cls.max_num_sample = 1000
        cls.max_num_points = 1000
        cls.degree_class = np.unique(np.logspace(0, 2, 30).astype("int"))

        # stationary and meanfield metrics
        cls.num_windows = 100
        cls.sample_graph = 0.05
        cls.num_samples = 25
        cls.burn = 50
        cls.adaptive = True
        cls.num_nodes = 2000
        cls.epsilon = 2e-3
        cls.full_data = False
        cls.parameters = np.concatenate(
            (np.linspace(0.1, 6.0, 50), np.linspace(6.1, 10, 10),)
        )
        # cls.parameters = np.linspace(1.0, 4.5, 50)

        cls.finder = "RecurrenceFPF"
        cls.num_k = 5
        cls.tol = 1e-6
        cls.max_iter = 5000
        cls.rec_iter = 100
        cls.initial_iter = 100
        cls.with_numba = True

        # summaries
        cls.axis = [1, 3]
        cls.transitions = [(0, 1), (1, 0)]
        cls.err_reduce = "percentile"

        return cls

    @classmethod
    def partially_hidden_sissis(cls):
        cls = cls()
        cls.names = []

        # ltp and statistcs metrics
        cls.max_num_sample = 1000
        cls.max_num_points = 1000
        cls.degree_class = np.unique(np.logspace(0, 2, 30).astype("int"))

        # stationary and meanfield metrics
        cls.num_windows = 100
        cls.sample_graph = 0.0
        cls.num_samples = 25
        cls.burn = 50
        cls.adaptive = True
        cls.num_nodes = 2000
        cls.epsilon = 2e-3
        cls.full_data = False
        cls.parameters = np.concatenate(
            (np.linspace(0.1, 6.0, 50), np.linspace(6.1, 10, 10),)
        )
        # cls.parameters = np.linspace(1.0, 4.5, 50)

        cls.finder = "RecurrenceFPF"
        cls.num_k = 5
        cls.tol = 1e-6
        cls.max_iter = 5000
        cls.rec_iter = 100
        cls.initial_iter = 100
        cls.with_numba = True

        # summaries
        cls.axis = [1, 3]
        cls.transitions = [(0, 1), (1, 0)]
        cls.err_reduce = "percentile"

        return cls

    @classmethod
    def dsis(cls):
        cls = cls()
        cls.names = []

        cls.max_num_points = 1e4
        cls.epsilon = 1e-3
        cls.adaptive = True
        cls.num_samples = 25
        cls.num_windows = 50
        cls.sample_graph = False
        cls.full_data = False
        cls.burn = 20
        cls.parameters = np.linspace(0, 1, 10)

        # forecast metrics
        cls.num_forecasts = 20
        cls.num_steps = 100

        return cls

    @classmethod
    def dsir(cls):
        cls = cls()
        cls.names = []

        cls.max_num_points = 1e6

        cls.epsilon = 1e-3
        cls.adaptive = True
        cls.num_samples = 25
        cls.num_windows = 50
        cls.sample_graph = False
        cls.full_data = False
        cls.burn = 20
        cls.parameters = np.linspace(0.01, 1, 50)

        # forecast metrics
        cls.num_forecasts = 20
        cls.num_steps = 100

        return cls

    @classmethod
    def rdsis(cls):
        cls = cls()
        cls.names = []

        cls.max_num_points = 1e6
        cls.epsilon = 1e-3
        cls.adaptive = True
        cls.num_samples = 25
        cls.num_windows = 50
        cls.sample_graph = False
        cls.full_data = False
        cls.burn = 20
        cls.parameters = np.linspace(0, 1, 10)

        # forecast metrics
        cls.num_forecasts = 20
        cls.num_steps = 100

        return cls

    @classmethod
    def rdsir(cls):
        cls = cls()
        cls.names = []

        cls.max_num_points = 1e6

        cls.epsilon = 1e-3
        cls.adaptive = True
        cls.num_samples = 25
        cls.num_windows = 50
        cls.sample_graph = False
        cls.full_data = False
        cls.burn = 20
        cls.parameters = np.linspace(0.01, 1, 50)

        # forecast metrics
        cls.num_forecasts = 20
        cls.num_steps = 100

        return cls

    @classmethod
    def rtn_forecast(cls):
        cls = cls()
        cls.names = [
            "TrueESSMetrics",
            "GNNESSMetrics",
            "TrueRTNForecastMetrics",
            "GNNRTNForecastMetrics",
        ]

        # stationary and meanfield metrics
        cls.num_samples = 100
        cls.epsilon = 1e-1
        cls.full_data = True
        cls.burn = 500

        # forecast metrics
        cls.num_forecasts = 10
        cls.num_steps = 10

        return cls
