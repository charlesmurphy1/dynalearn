from .forecast import *
from .ltp import *
from .jsderror import *
from .meanfield import *
from .stationary import *
from .statistics import *

__summaries__ = {
    "TrueLTPMetrics": TrueLTPSummary,
    "GNNLTPMetrics": GNNLTPSummary,
    "MLELTPMetrics": MLELTPSummary,
    "TrueStarLTPMetrics": TrueStarLTPSummary,
    "GNNStarLTPMetrics": GNNStarLTPSummary,
    "UniformStarLTPMetrics": UniformStarLTPSummary,
    "StatisticsMetrics": StatisticsSummary,
    "TrueSSMetrics": TrueSSSummary,
    "GNNSSMetrics": GNNSSSummary,
    "TrueESSMetrics": TrueESSSummary,
    "GNNESSMetrics": GNNESSSummary,
    "TruePESSMetrics": TruePESSSummary,
    "GNNPESSMetrics": GNNPESSSummary,
    "TruePEMFMetrics": TruePEMFSummary,
    "GNNPEMFMetrics": GNNPEMFSummary,
    "TrueForecastMetrics": TrueForecastSummary,
    "GNNForecastMetrics": GNNForecastSummary,
    "TrueRTNForecastMetrics": TrueRTNForecastSummary,
    "GNNRTNForecastMetrics": GNNRTNForecastSummary,
}


def get(config):
    names = config.names
    summaries = {}
    for n in names:
        if n in __summaries__:
            summaries[n] = __summaries__[n](config)
            if n == "TrueLTPMetrics":
                summaries["JSDErrorSummary"] = JSDErrorSummary(config)
            if n == "TrueStarLTPMetrics":
                summaries["StarJSDErrorSummary"] = StarJSDErrorSummary(config)
        else:
            raise ValueError(
                f"{n} is invalid, possible entries are {list(__summaries__.keys())}"
            )
    return summaries
