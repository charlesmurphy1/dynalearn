import dynalearn as dl

aggregators = {
    "SIS": dl.metrics.SimpleContagionAggregator,
    "SIR": dl.metrics.SimpleContagionAggregator,
    "NonLinearSIS": dl.metrics.SimpleContagionAggregator,
    "NonLinearSIR": dl.metrics.SimpleContagionAggregator,
    "SineSIS": dl.metrics.SimpleContagionAggregator,
    "SineSIR": dl.metrics.SimpleContagionAggregator,
    "PlanckSIS": dl.metrics.SimpleContagionAggregator,
    "PlanckSIR": dl.metrics.SimpleContagionAggregator,
    "SoftThresholdSIS": dl.metrics.ComplexContagionAggregator,
    "SoftThresholdSIR": dl.metrics.ComplexContagionAggregator,
    "SISSIS": dl.metrics.InteractingContagionAggregator,
}

metrics_dict = {
    "StatisticsMetrics": dl.metrics.StatisticsMetrics,
    "TrueLTPMetrics": dl.metrics.TrueLTPMetrics,
    "GNNLTPMetrics": dl.metrics.GNNLTPMetrics,
    "MLELTPMetrics": dl.metrics.MLELTPMetrics,
    "TrueStarLTPMetrics": dl.metrics.TrueStarLTPMetrics,
    "GNNStarLTPMetrics": dl.metrics.GNNStarLTPMetrics,
    "AttentionMetrics": dl.metrics.AttentionMetrics,
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

    if name in aggregators:
        return aggregators[name]()
    else:
        raise ValueError(
            "Wrong name of dynamics. Valid entries are: {0}".format(
                list(aggregators.keys())
            )
        )
    for name in metrics_names:
        if m in metrics_dict:
            metrics[name] = metrics_dict[name](aggregator=aggregator)
        else:
            raise ValueError(
                "Wrong name of metrics. Valid entries are: {0}".format(
                    list(metrics_dict.keys())
                )
            )

    return metrics
