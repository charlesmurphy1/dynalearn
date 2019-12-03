import numpy as np


class Aggregator(object):
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
    ):

        y = np.zeros(x.shape)
        err = np.zeros(x.shape)

        if operation == "mean":
            op_val = np.nanmean
            op_err = lambda xx: np.sqrt(np.nanvar(xx))
        elif operation == "sum":
            op_val = np.nansum
            op_err = lambda x: 0

        if for_degree:
            x = np.unique(np.sort(np.sum(summaries[:, 1:], axis=-1)))
            x = x[x > 0]
            for i, xx in enumerate(x):

                if in_state is None:
                    index = np.sum(summaries[:, 1:], axis=-1) == xx
                else:
                    index = (np.sum(summaries[:, 1:], axis=-1) == xx) * (
                        summaries[:, 0] == in_state
                    )

                if out_state is None:
                    y[i] = op_val(values[index])
                    err[i] = op_err(values[index])
                else:
                    y[i] = op_val(values[index, out_state])
                    err[i] = op_err(values[index, out_state])
        else:
            x, all_x = self.aggregate_summaries(summaries)
            for i, xx in enumerate(x):

                if in_state is None:
                    index = all_x == xx
                else:
                    index = (all_x == xx) * (summaries[:, 0] == in_state)

                if out_state is None:
                    y[i] = op_val(values[index])
                    err[i] = op_err(values[index])
                else:
                    y[i] = op_val(values[index, out_state])
                    err[i] = op_err(values[index, out_state])

        x = x[~np.isnan(y)]
        y = y[~np.isnan(y)]
        err = err[~np.isnan(y)]
        return x, y, err

    def aggregate_summaries(self, summaries):
        raise NotImplementedError()


class SimpleContagionAggregator(Aggregator):
    def __init__(self):
        super(SimpleContagionAggregator, self).__init__()

    def aggregate_summaries(self, summaries):
        sorted_val = np.unique(np.sort(summaries[:, 2]))

        all_val = summaries[:, 2]
        return sorted_val, all_val


class InteractingContagionAggregator(Aggregator):
    def __init__(self, agent):
        super(InteractingContagionAggregator, self).__init__()
        self.agent = agent

    def aggregate_summaries(self, summaries):
        sorted_val = np.unique(np.sort(summaries[:, self.agent + 2] + summaries[:, -1]))
        all_val = summaries[:, self.agent + 2] + summaries[:, -1]
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
