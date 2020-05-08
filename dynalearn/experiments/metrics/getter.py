from .forecast import *
from .ltp import *
from .starltp import *
from .statistics import *
from .stationary import *
from .meanfield import *

__metrics__ = {
    "TrueLTPMetrics": TrueLTPMetrics,
    "TrueStarLTPMetrics": TrueStarLTPMetrics,
    "GNNLTPMetrics": GNNLTPMetrics,
    "GNNStarLTPMetrics": GNNStarLTPMetrics,
    "MLELTPMetrics": MLELTPMetrics,
    "UniformStarLTPMetrics": UniformStarLTPMetrics,
    "StatisticsMetrics": StatisticsMetrics,
    "TrueSSMetrics": TrueSSMetrics,
    "GNNSSMetrics": GNNSSMetrics,
    "TruePESSMetrics": TruePESSMetrics,
    "GNNPESSMetrics": GNNPESSMetrics,
    "TruePEMFMetrics": TruePEMFMetrics,
    "GNNPEMFMetrics": GNNPEMFMetrics,
    "TrueForecastMetrics": TrueForecastMetrics,
    "GNNForecastMetrics": GNNForecastMetrics,
    "TrueRTNForecastMetrics": TrueRTNForecastMetrics,
    "GNNRTNForecastMetrics": GNNRTNForecastMetrics,
}


def get(config):
    names = config.names
    metrics = {}
    for n in names:
        if n in __metrics__:
            metrics[n] = __metrics__[n](config)
        else:
            raise ValueError(
                f"{n} is invalid, possible entries are {list(__metrics__.keys())}"
            )
    return metrics
