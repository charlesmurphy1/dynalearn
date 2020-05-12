from .forecast import *
from .ltp import *
from .jsderror import *
from .meanfield import *
from .stationary import *
from .statistics import *

__summaries__ = {
    "TrueLTPSummary": TrueLTPSummary,
    "GNNLTPSummary": GNNLTPSummary,
    "MLELTPSummary": MLELTPSummary,
    "TrueStarLTPSummary": TrueStarLTPSummary,
    "GNNStarLTPSummary": GNNStarLTPSummary,
    "UniformStarLTPSummary": UniformStarLTPSummary,
    "TrueGNNJSDErrorSummary": TrueGNNJSDErrorSummary,
    "TrueMLEJSDErrorSummary": TrueMLEJSDErrorSummary,
    "TrueGNNStarJSDErrorSummary": TrueGNNStarJSDErrorSummary,
    "TrueUniformStarJSDErrorSummary": TrueUniformStarJSDErrorSummary,
    "StatisticsSummary": StatisticsSummary,
    "TrueSSSummary": TrueSSSummary,
    "GNNSSSummary": GNNSSSummary,
    "TrueESSSummary": TrueESSSummary,
    "GNNESSSummary": GNNESSSummary,
    "TruePESSSummary": TruePESSSummary,
    "GNNPESSSummary": GNNPESSSummary,
    "TruePEMFSummary": TruePEMFSummary,
    "GNNPEMFSummary": GNNPEMFSummary,
    "TrueForecastSummary": TrueForecastSummary,
    "GNNForecastSummary": GNNForecastSummary,
    "TrueRTNForecastSummary": TrueRTNForecastSummary,
    "GNNRTNForecastSummary": GNNRTNForecastSummary,
}


def get(config):
    names = config.names
    summaries = {}
    for n in names:
        if n in __summaries__:
            summaries[n] = __summaries__[n](config)
        else:
            raise ValueError(
                f"{n} is invalid, possible entries are {list(__summaries__.keys())}"
            )
    return summaries
