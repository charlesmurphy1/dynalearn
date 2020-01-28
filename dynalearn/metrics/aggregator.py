import numpy as np
from abc import ABC


def aggregate_std(x):
    err = std(x)
    err[np.isnan(err)] = 0
    ret


class Aggregator(ABC):
    def __init__(self):
        super(Aggregator, self).__init__()

    def __call__(
        self,
        summaries,
        values,
        in_state=None,
        out_state=None,
        for_degree=False,
        operation="mean",
        err_operation="std",
    ):

        if operation == "mean":
            op_val = np.nanmean
            if err_operation == "std":
                op_err = lambda xx: (
                    np.nanmean(xx) - np.nanstd(xx),
                    np.nanmean(xx) + np.nanstd(xx),
                )
            elif err_operation == "percentile":
                op_err = lambda xx: (np.nanpercentile(xx, 16), np.nanpercentile(xx, 84))
        elif operation == "sum":
            op_val = np.nansum
            op_err = lambda x: (0, 0)

        if for_degree:
            x = np.unique(np.sort(np.sum(summaries[:, 1:], axis=-1)))
            x = x[x > 0]
            y = np.zeros(x.shape)
            err_low = np.zeros(x.shape)
            err_high = np.zeros(x.shape)
            for i, xx in enumerate(x):

                if in_state is None:
                    index = np.sum(summaries[:, 1:], axis=-1) == xx
                else:
                    index = (np.sum(summaries[:, 1:], axis=-1) == xx) * (
                        summaries[:, 0] == in_state
                    )

                if out_state is None:
                    val = values[index]
                else:
                    val = values[index, out_state]

                if len(val) > 0:
                    y[i] = op_val(val)
                    err_low[i], err_high[i] = op_err(val)
                else:
                    y[i] = np.nan
        else:
            x, all_x = self.aggregate_summaries(summaries)
            y = np.zeros(x.shape)
            err_low = np.zeros(x.shape)
            err_high = np.zeros(x.shape)
            for i, xx in enumerate(x):

                if in_state is None:
                    index = all_x == xx
                else:
                    index = (all_x == xx) * (summaries[:, 0] == in_state)

                if out_state is None:
                    val = values[index]
                else:
                    val = values[index, out_state]

                if len(val) > 0:
                    y[i] = op_val(val)
                    err_low[i], err_high[i] = op_err(val)
                else:
                    y[i] = np.nan

        x = x[~np.isnan(y)]
        err_low = err_low[~np.isnan(y)]
        err_high = err_high[~np.isnan(y)]
        y = y[~np.isnan(y)]
        return x, y, err_low, err_high

    def aggregate_summaries(self, summaries):
        raise NotImplementedError("aggregate_summaries must be implemented.")


class SimpleContagionAggregator(Aggregator):
    def __init__(self):
        super(SimpleContagionAggregator, self).__init__()

    def aggregate_summaries(self, summaries):
        sorted_val = np.unique(np.sort(summaries[:, 2]))

        all_val = summaries[:, 2]
        return sorted_val, all_val


class InteractingContagionAggregator(Aggregator):
    def __init__(self):
        super(InteractingContagionAggregator, self).__init__()

    def aggregate_summaries(self, summaries):
        sorted_val = np.unique(np.sort(np.sum(summaries[:, 1:], axis=-1)))
        all_val = np.sum(summaries[:, 1:], axis=-1)
        return sorted_val, all_val


class ComplexContagionAggregator(Aggregator):
    def __init__(self):
        super(ComplexContagionAggregator, self).__init__()

    def aggregate_summaries(self, summaries):
        k = np.sum(summaries[:, 1:], axis=1)
        l = summaries[:, 2]
        sorted_val = np.sort(np.unique(l[k > 0] / k[k > 0]))
        all_val = l / k
        return sorted_val, all_val
