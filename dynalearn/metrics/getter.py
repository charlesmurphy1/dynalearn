import dynalearn as dl


def get_aggregator(dynamics):
    if "SIS" == type(dynamics).__name__:
        return dl.metrics.SimpleContagionAggregator()
    elif "SIR" == type(dynamics).__name__:
        return dl.metrics.SimpleContagionAggregator()
    elif "SoftThresholdSIS" == type(dynamics).__name__:
        return dl.metrics.ComplexContagionAggregator()
    elif "SoftThresholdSIR" == type(dynamics).__name__:
        return dl.metrics.ComplexContagionAggregator()
    elif "NonLinearSIS" == type(dynamics).__name__:
        return dl.metrics.SimpleContagionAggregator()
    elif "NonLinearSIR" == type(dynamics).__name__:
        return dl.metrics.SimpleContagionAggregator()
    elif "SineSIS" == type(dynamics).__name__:
        return dl.metrics.SimpleContagionAggregator()
    elif "SineSIR" == type(dynamics).__name__:
        return dl.metrics.SimpleContagionAggregator()
    elif "PlanckSIS" == type(dynamics).__name__:
        return dl.metrics.SimpleContagionAggregator()
    elif "PlanckSIR" == type(dynamics).__name__:
        return dl.metrics.SimpleContagionAggregator()
    elif (
        "SISSIS" == type(dynamics).__name__
        or "CooperativeContagionSIS" == type(dynamics).__name__
    ):
        return dl.utilities.InteractingContagionAggregator(0)
    else:
        raise ValueError("wrong name of aggregator.")


def get(metrics_names, dynamics):
    aggregator = get_aggregator(dynamics)
    metrics = {}
    for m in metrics_names:
        if m == "StatisticsMetrics":
            metrics[m] = dl.metrics.StatisticsMetrics(aggregator=aggregator)
        elif m == "TrueLTPMetrics":
            metrics[m] = dl.metrics.TrueLTPMetrics(aggregator=aggregator)
        elif m == "GNNLTPMetrics":
            metrics[m] = dl.metrics.GNNLTPMetrics(aggregator=aggregator)
        elif m == "MLELTPMetrics":
            metrics[m] = dl.metrics.MLELTPMetrics(aggregator=aggregator)
        elif m == "TrueStarLTPMetrics":
            metrics[m] = dl.metrics.TrueStarLTPMetrics(aggregator=aggregator)
        elif m == "GNNStarLTPMetrics":
            metrics[m] = dl.metrics.GNNStarLTPMetrics(aggregator=aggregator)
        elif m == "AttentionMetrics":
            metrics[m] = dl.metrics.AttentionMetrics()
        else:
            raise ValueError("wrong name of metric.")

    return metrics
