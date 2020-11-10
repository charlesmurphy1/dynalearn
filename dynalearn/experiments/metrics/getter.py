from .metrics import CustomMetrics
from .ltp import *
from .prediction import *
from .starltp import *
from .statistics import *
from .stationary import *

__metrics__ = {
    "TrueLTPMetrics": TrueLTPMetrics,
    "TrueStarLTPMetrics": TrueStarLTPMetrics,
    "GNNLTPMetrics": GNNLTPMetrics,
    "GNNStarLTPMetrics": GNNStarLTPMetrics,
    "MLELTPMetrics": MLELTPMetrics,
    "UniformStarLTPMetrics": UniformStarLTPMetrics,
    "PredictionMetrics": PredictionMetrics,
    "StatisticsMetrics": StatisticsMetrics,
    "TruePSSMetrics": TruePSSMetrics,
    "GNNPSSMetrics": GNNPSSMetrics,
    "TrueERSSMetrics": TrueERSSMetrics,
    "GNNERSSMetrics": GNNERSSMetrics,
}


def get(config):
    names = config.names
    metrics = {}
    for n in names:
        if n in __metrics__:
            metrics[n] = __metrics__[n](config)
        else:
            metrics[n] = CustomMetrics(config)
            # raise ValueError(
            #     f"{n} is invalid, possible entries are {list(__metrics__.keys())}"
            # )
    return metrics
