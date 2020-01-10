from .base import Metrics
import matplotlib.pyplot as plt
import numpy as np
from random import sample
from scipy.special import binom
from scipy.spatial.distance import jensenshannon
import tqdm
from abc import abstractmethod

max_num_sample = 1000000


def all_combinations(k, d):
    if d == 1:
        return [[k]]
    return [(*j, k - i) for i in range(k + 1) for j in all_combinations(i, d - 1)]


class StarLTPMetrics(Metrics):
    def __init__(self, config, verbose=1):
        self.__config = config
        self.degree_class = config.degree_class
        self.aggregator = config.aggregator
        super(StarLTPMetrics, self).__init__(verbose)

    @abstractmethod
    def get_metric(self, experiment, input, adj):
        raise NotImplementedError("get_metric must be implemented.")

    def aggregate(
        self,
        data=None,
        in_state=None,
        out_state=None,
        for_degree=False,
        err_operation="std",
    ):
        if self.aggregate is None:
            return

        if data is None:
            data = self.data["ltp"]

        x, y, err_low, err_high = self.aggregator(
            self.data["summaries"],
            data,
            in_state=in_state,
            out_state=out_state,
            for_degree=for_degree,
            operation="mean",
            err_operation=err_operation,
        )
        return x, y, err_low, err_high

    def compare(self, name, metrics, func=None, verbose=False):
        size = self.data["summaries"].shape[0]
        if verbose:
            pbar = tqdm.tqdm(range(size), f"Comparing using {func}")
        ans = np.zeros(size)

        if np.any(self.data["summaries"] != metrics.data["summaries"]):
            ValueError("Summaries are not the same.")
        for i, s in enumerate(self.data["summaries"]):
            ans[i] = func(self.data["ltp"][i], metrics.data[name][i])
            if verbose:
                pbar.update()
        if verbose:
            pbar.close()
        return ans

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
    def __init__(self, config, verbose=1):
        super(TrueStarLTPMetrics, self).__init__(config, verbose)

    def get_metric(self, experiment, inputs, adj):
        return experiment.dynamics_model.predict(inputs, adj)[0]


class GNNStarLTPMetrics(StarLTPMetrics):
    def __init__(self, config, verbose=1):
        super(GNNStarLTPMetrics, self).__init__(config, verbose)

    def get_metric(self, experiment, inputs, adj):
        return experiment.model.predict(inputs, adj)[0]


class UniformStarLTPMetrics(StarLTPMetrics):
    def __init__(self, config, verbose=1):
        super(UniformStarLTPMetrics, self).__init__(config, verbose)

    def get_metric(self, experiment, inputs, adj):
        num_states = len(experiment.dynamics_model.state_label)
        return np.ones(num_states) / num_states
