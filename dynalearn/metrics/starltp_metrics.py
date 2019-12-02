from .base_metrics import Metrics
import matplotlib.pyplot as plt
import numpy as np
from random import sample
from scipy.special import binom
from scipy.spatial.distance import jensenshannon
import tqdm

max_num_sample = 1000000


def all_combinations(k, d):
    if d == 1:
        return [[k]]
    return [(*j, k - i) for i in range(k + 1) for j in all_combinations(i, d - 1)]


class StarLTPMetrics(Metrics):
    def __init__(self, degree_class=None, aggregator=None, verbose=1):
        if degree_class is None:
            self.degree_class = np.unique(np.logspace(0, 2, 30).astype("int"))
        else:
            self.degree_class = degree_class
        self.aggregator = aggregator
        super(StarLTPMetrics, self).__init__(verbose)

    def get_metric(self, experiment, input, adj):
        raise NotImplementedError()

    def aggregate(self, in_state=None, out_state=None, dataset="train"):
        if self.aggregate is None:
            return
        x, y, err = self.aggregator(
            self.data["summaries"],
            self.data["ltp/" + dataset],
            in_state=in_state,
            out_state=out_state,
            operation="mean",
        )
        return x, y, err

    def display(
        self, in_state, out_state, num_points=None, ax=None, fill=None, **plot_kwargs
    ):
        if ax is None:
            ax = plt.gca()
        if "ltp" not in self.data or self.aggregator is None:
            return ax

        x, y, err = self.aggregator(
            self.data["summaries"],
            self.data["ltp"],
            in_state=in_state,
            out_state=out_state,
            operation="mean",
        )
        if num_points is not None and len(x) > num_points:
            w = round(len(x) / num_points)
            x = x[::w]
            y = y[::w]
            err = err[::w]
        if fill is not None:
            ax.fill_between(x, y - err, y + err, color=fill, alpha=0.3)
        ax.plot(x, y, **plot_kwargs)
        return ax

    def compute(self, experiment):

        N = max(self.degree_class) + 1
        prev_N = experiment.model.num_nodes
        experiment.model.num_nodes = N

        state_label = experiment.dynamics_model.state_label
        d = len(state_label)
        summaries = {}

        if self.verbose:
            num_iter = int(
                d
                * np.sum(
                    [
                        binom(k + d - 1, d - 1)
                        if binom(k + d - 1, d - 1) < max_num_sample
                        else max_num_sample
                        for k in self.degree_class
                    ]
                )
            )
            p_bar = tqdm.tqdm(range(num_iter), "Computing " + self.__class__.__name__)

        for k in self.degree_class:
            adj = np.zeros((N, N))
            adj[1 : k + 1, 0] = 1
            adj[0, 1 : k + 1] = 1
            all_comb = all_combinations(k, len(state_label))
            if len(all_comb) > max_num_sample:
                all_comb = sample(all_comb, max_num_sample)
            for s in all_comb:
                neighbors_states = np.concatenate(
                    [i * np.ones(l) for i, l in enumerate(s)]
                )

                inputs = np.zeros(max(self.degree_class) + 1)
                inputs[1 : k + 1] = neighbors_states
                for in_s, in_l in state_label.items():
                    inputs[0] = in_l
                    summaries[(in_l, *s)] = self.get_metric(experiment, inputs, adj)
                    if self.verbose:
                        p_bar.update()

        if self.verbose:
            p_bar.close()

        self.data["summaries"] = np.array([s for s in summaries])
        self.data["ltp"] = np.array(
            [summaries[tuple(s)] for s in self.data["summaries"]]
        )

        experiment.model.num_nodes = prev_N


class TrueStarLTPMetrics(StarLTPMetrics):
    def __init__(self, degree_class=None, aggregator=None, verbose=1):
        super(TrueStarLTPMetrics, self).__init__(degree_class, aggregator, verbose)

    def get_metric(self, experiment, inputs, adj):
        return experiment.dynamics_model.predict(inputs, adj)[0]


class GNNStarLTPMetrics(StarLTPMetrics):
    def __init__(self, degree_class=None, aggregator=None, verbose=1):
        super(GNNStarLTPMetrics, self).__init__(degree_class, aggregator, verbose)

    def get_metric(self, experiment, inputs, adj):
        return experiment.model.predict(inputs, adj)[0]
