import numpy as np


class Aggregator(object):
    def __init__(self):
        super(Aggregator, self).__init__()

    def __call__(
        self, in_state, summaries, mean, var=None, out_state=None, operation="mean"
    ):

        x, all_x = self.aggregate_summaries(summaries)
        y = np.zeros(x.shape)
        err = np.zeros(x.shape)

        if operation == "mean":
            operation_on_mean = np.nanmean
            operation_on_var = lambda var: np.nanmean(var) / var[~np.isnan(var)].size
        elif operation == "sum":
            operation_on_mean = np.nansum
            operation_on_var = np.nansum
        for i, xx in enumerate(x):
            if in_state is None:
                index = all_x == xx
            else:
                index = (all_x == xx) * (summaries[:, 0] == in_state)
            if out_state is None:
                y[i] = operation_on_mean(mean[index])
            else:
                y[i] = operation_on_mean(mean[index, out_state])
            if var is not None:
                if out_state is None:
                    err[i] = np.sqrt(operation_on_var(var[index]))
                else:
                    err[i] = np.sqrt(operation_on_var(var[index, out_state]))
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
