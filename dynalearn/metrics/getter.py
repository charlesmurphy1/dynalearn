from .aggregator import *
from .base_metrics import *
from .ltp_metrics import *
from .starltp_metrics import *
from .attention_metrics import *
from .loss_metrics import *
from .statistics_metrics import *
from .meanfield_metrics import *
from .stationarystate_metrics import *

aggregators = {
    "SIS": SimpleContagionAggregator,
    "SIR": SimpleContagionAggregator,
    "NonLinearSIS": SimpleContagionAggregator,
    "NonLinearSIR": SimpleContagionAggregator,
    "SineSIS": SimpleContagionAggregator,
    "SineSIR": SimpleContagionAggregator,
    "PlanckSIS": SimpleContagionAggregator,
    "PlanckSIR": SimpleContagionAggregator,
    "SoftThresholdSIS": ComplexContagionAggregator,
    "SoftThresholdSIR": ComplexContagionAggregator,
    "SISSIS": InteractingContagionAggregator,
}

metrics_dict = {
    "StatisticsMetrics": (StatisticsMetrics, True),
    "TrueLTPMetrics": (TrueLTPMetrics, True),
    "GNNLTPMetrics": (GNNLTPMetrics, True),
    "MLELTPMetrics": (MLELTPMetrics, True),
    "TrueStarLTPMetrics": (TrueStarLTPMetrics, True),
    "GNNStarLTPMetrics": (GNNStarLTPMetrics, True),
    "AttentionMetrics": (AttentionMetrics, False),
    "PoissonEpidemicsSSMetrics": (PoissonEpidemicsSSMetrics, False),
    "PoissonEpidemicsMFMetrics": (PoissonEpidemicsMFMetrics, False),
}


def get_aggregator(dynamics):
    name = type(dynamics).__name__

    if name in aggregators:
        return aggregators[name]()
    else:
        raise ValueError(
            "Wrong name of dynamics for aggregator. Valid entries are: {0}".format(
                list(aggregators.keys())
            )
        )


def get(metrics_names, dynamics):
    aggregator = get_aggregator(dynamics)
    metrics = {}
    for name in metrics_names:
        if name in metrics_dict:
            if metrics_dict[name][1]:
                metrics[name] = metrics_dict[name][0](aggregator=aggregator)
            else:
                metrics[name] = metrics_dict[name][0]()
        else:
            raise ValueError(
                "Wrong name of metrics. Valid entries are: {0}".format(
                    list(metrics_dict.keys())
                )
            )

    return metrics
