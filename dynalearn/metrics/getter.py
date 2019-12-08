from .ltp_metrics import *
from .starltp_metrics import *
from .attention_metrics import *
from .loss_metrics import *
from .statistics_metrics import *
from .meanfield_metrics import *
from .stationarystate_metrics import *

metrics_dict = {
    "StatisticsMetrics": StatisticsMetrics,
    "TrueLTPMetrics": TrueLTPMetrics,
    "GNNLTPMetrics": GNNLTPMetrics,
    "MLELTPMetrics": MLELTPMetrics,
    "TrueStarLTPMetrics": TrueStarLTPMetrics,
    "GNNStarLTPMetrics": GNNStarLTPMetrics,
    "AttentionMetrics": AttentionMetrics,
    "PoissonEpidemicsSSMetrics": PoissonEpidemicsSSMetrics,
    "PoissonEpidemicsMFMetrics": PoissonEpidemicsMFMetrics,
}


def get(param_dict, dynamics):
    names = param_dict["name"]
    config = param_dict["config"]
    metrics = {}
    for n in names:
        if n in metrics_dict:
            metrics[n] = metrics_dict[n](config)
        else:
            raise ValueError(
                "Wrong name of metrics. Valid entries are: {0}".format(
                    list(metrics_dict.keys())
                )
            )

    return metrics
