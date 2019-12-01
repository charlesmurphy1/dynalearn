from dynalearn.metrics import *


def get_aggregator(dynamics):
    if "SIS" == type(dynamics).__name__:
        return dl.utilities.SimpleContagionAggregator()
    elif "SIR" == type(dynamics).__name__:
        return dl.utilities.SimpleContagionAggregator()
    elif "SoftThresholdSIS" == type(dynamics).__name__:
        return dl.utilities.ComplexContagionAggregator()
    elif "SoftThresholdSIR" == type(dynamics).__name__:
        return dl.utilities.ComplexContagionAggregator()
    elif "NonLinearSIS" == type(dynamics).__name__:
        return dl.utilities.SimpleContagionAggregator()
    elif "NonLinearSIR" == type(dynamics).__name__:
        return dl.utilities.SimpleContagionAggregator()
    elif "SineSIS" == type(dynamics).__name__:
        return dl.utilities.SimpleContagionAggregator()
    elif "SineSIR" == type(dynamics).__name__:
        return dl.utilities.SimpleContagionAggregator()
    elif "PlanckSIS" == type(dynamics).__name__:
        return dl.utilities.SimpleContagionAggregator()
    elif "PlanckSIR" == type(dynamics).__name__:
        return dl.utilities.SimpleContagionAggregator()
    elif (
        "SISSIS" == type(dynamics).__name__
        or "CooperativeContagionSIS" == type(dynamics).__name__
    ):
        return dl.utilities.InteractingContagionAggregator(0)
    else:
        raise ValueError("wrong string name for aggregator.")


def get(metrics_names, dynamics):
    aggregator = get_aggregator(dynamics)
    metrics = {}
    for m in metrics_names:
        if m == "StatiticsMetrics":
            metrics[m] = StatiticsMetrics(aggregator=aggregator)
        elif m == "LTPMetrics":
            metrics[m] = LTPMetrics(aggregator=aggregator)
        elif m == "TrueStarLTPMetrics":
            metrics[m] = TrueStarLTPMetrics(aggregator=aggregator)
        elif m == "GNNStarLTPMetrics":
            metrics[m] = GNNStarLTPMetrics(aggregator=aggregator)
        elif m == "AttentionMetrics":
            metrics[m] = AttentionMetrics()
    return metrics
