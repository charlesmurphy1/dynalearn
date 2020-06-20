import numpy as np
from .config import Config


class MetricsConfig(Config):
    @classmethod
    def sis_fast(cls):
        cls = cls()
        cls.names = [
            "TrueLTPMetrics",
            "GNNLTPMetrics",
            "MLELTPMetrics",
            "TrueStarLTPMetrics",
            "GNNStarLTPMetrics",
            "UniformStarLTPMetrics",
            "StatisticsMetrics",
        ]

        # ltp and statistcs metrics
        cls.max_num_sample = 1000
        cls.max_num_points = 1000
        cls.degree_class = np.unique(np.logspace(0, 2, 30).astype("int"))

        return cls

    @classmethod
    def sis_complete(cls):
        cls = cls()
        cls.names = [
            "TrueLTPMetrics",
            "GNNLTPMetrics",
            "MLELTPMetrics",
            "TrueStarLTPMetrics",
            "GNNStarLTPMetrics",
            "UniformStarLTPMetrics",
            "StatisticsMetrics",
            "TruePESSMetrics",
            "GNNPESSMetrics",
            "TruePEMFMetrics",
            "GNNPEMFMetrics",
        ]

        # ltp and statistcs metrics
        cls.max_num_sample = 1000
        cls.max_num_points = 1000
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

        return cls

    @classmethod
    def plancksis_fast(cls):
        cls = cls()
        cls.names = [
            "TrueLTPMetrics",
            "GNNLTPMetrics",
            "MLELTPMetrics",
            "TrueStarLTPMetrics",
            "GNNStarLTPMetrics",
            "UniformStarLTPMetrics",
            "StatisticsMetrics",
        ]

        # ltp and statistcs metrics
        cls.max_num_sample = 1000
        cls.max_num_points = 1000
        cls.degree_class = np.unique(np.logspace(0, 2, 30).astype("int"))

        return cls

    @classmethod
    def plancksis_complete(cls):
        cls = cls()
        cls.names = [
            "TrueLTPMetrics",
            "GNNLTPMetrics",
            "MLELTPMetrics",
            "TrueStarLTPMetrics",
            "GNNStarLTPMetrics",
            "UniformStarLTPMetrics",
            "StatisticsMetrics",
            "TruePESSMetrics",
            "GNNPESSMetrics",
            "TruePEMFMetrics",
            "GNNPEMFMetrics",
        ]

        # ltp and statistcs metrics
        cls.max_num_sample = 1000
        cls.max_num_points = 1000
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
        # cls.parameters = np.linspace(3.25, 4.5, 50)

        cls.finder = "RecurrenceFPF"
        cls.num_k = 7
        cls.tol = 1e-6
        cls.max_iter = 5000
        cls.rec_iter = 100
        cls.initial_iter = 100
        cls.with_numba = True

        return cls

    @classmethod
    def sissis_fast(cls):
        cls = cls()
        cls.names = [
            "TrueLTPMetrics",
            "GNNLTPMetrics",
            "MLELTPMetrics",
            "TrueStarLTPMetrics",
            "GNNStarLTPMetrics",
            "UniformStarLTPMetrics",
            "StatisticsMetrics",
        ]

        # ltp and statistcs metrics
        cls.max_num_sample = 1000
        cls.max_num_points = 1000
        cls.degree_class = np.unique(np.logspace(0, 2, 30).astype("int"))

        return cls

    @classmethod
    def sissis_complete(cls):
        cls = cls()
        cls.names = [
            "TrueLTPMetrics",
            "GNNLTPMetrics",
            "MLELTPMetrics",
            "TrueStarLTPMetrics",
            "GNNStarLTPMetrics",
            "UniformStarLTPMetrics",
            "StatisticsMetrics",
            "TruePESSMetrics",
            "GNNPESSMetrics",
            "TruePEMFMetrics",
            "GNNPEMFMetrics",
        ]

        # ltp and statistcs metrics
        cls.max_num_sample = 1000
        cls.max_num_points = 1000
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
                np.linspace(0.1, 1.0, 5),
                np.linspace(1.0, 4.5, 50),
                np.linspace(4.6, 7, 10),
            )
        )
        # cls.parameters = np.linspace(1.0, 4.5, 50)

        cls.finder = "RecurrenceFPF"
        cls.num_k = 7
        cls.tol = 1e-6
        cls.max_iter = 5000
        cls.rec_iter = 100
        cls.initial_iter = 100
        cls.with_numba = True

        return cls

    @classmethod
    def hidden_sissis_fast(cls):
        cls = cls()
        cls.names = [
            "TrueLTPMetrics",
            "GNNLTPMetrics",
            "MLELTPMetrics",
            # "TrueStarLTPMetrics",
            # "GNNStarLTPMetrics",
            # "UniformStarLTPMetrics",
            "StatisticsMetrics",
        ]

        # ltp and statistcs metrics
        cls.max_num_sample = 1000
        cls.max_num_points = 1000
        cls.degree_class = np.unique(np.logspace(0, 2, 30).astype("int"))

        return cls

    @classmethod
    def hidden_sissis_complete(cls):
        cls = cls()
        cls.names = [
            "TrueLTPMetrics",
            "GNNLTPMetrics",
            "MLELTPMetrics",
            # "TrueStarLTPMetrics",
            # "GNNStarLTPMetrics",
            # "UniformStarLTPMetrics",
            "StatisticsMetrics",
            "TruePESSMetrics",
            "GNNPESSMetrics",
            "TruePEMFMetrics",
            "GNNPEMFMetrics",
        ]

        # ltp and statistcs metrics
        cls.max_num_sample = 1000
        cls.max_num_points = 1000
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
                np.linspace(0.1, 1.0, 5),
                np.linspace(1.0, 4.5, 50),
                np.linspace(4.6, 7, 10),
            )
        )
        # cls.parameters = np.linspace(1.0, 4.5, 50)

        cls.finder = "RecurrenceFPF"
        cls.num_k = 7
        cls.tol = 1e-6
        cls.max_iter = 5000
        cls.rec_iter = 100
        cls.initial_iter = 100
        cls.with_numba = True

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

    @classmethod
    def test(cls):
        cls = cls()
        cls.names = [
            "TrueLTPMetrics",
            "GNNLTPMetrics",
            "MLELTPMetrics",
            "TrueStarLTPMetrics",
            "GNNStarLTPMetrics",
            "UniformStarLTPMetrics",
            "StatisticsMetrics",
            "TruePESSMetrics",
            "GNNPESSMetrics",
            "TruePEMFMetrics",
            "GNNPEMFMetrics",
        ]

        # ltp and statistcs metrics
        cls.max_num_points = 1000
        cls.max_num_sample = 1000
        cls.degree_class = np.unique(np.logspace(0, 2, 10).astype("int"))

        # stationary and meanfield metrics
        cls.num_samples = 10
        cls.num_nodes = 100
        cls.epsilon = 1e-2
        cls.full_data = False
        cls.burn = 500
        cls.parameters = np.linspace(0.01, 10.0, 10)

        cls.finder = "RecurrenceFPF"
        cls.num_k = 7
        cls.tol = 1e-6
        cls.max_iter = 5000
        cls.rec_iter = 100
        cls.initial_iter = 100
        cls.with_numba = True

        return cls
