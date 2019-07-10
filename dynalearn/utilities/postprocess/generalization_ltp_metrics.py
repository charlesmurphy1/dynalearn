from .base_metrics import Metrics
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import binom
from scipy.spatial.distance import jensenshannon
import tqdm


def all_combinations(k, d):
    if d == 1:
        return [[k]]
    return [(*j, k - i) for i in range(k + 1) for j in all_combinations(i, d - 1)]


class LTPGeneralizationMetrics(Metrics):
    def __init__(self, degree_class, verbose=1):
        super(LTPGeneralizationMetrics, self).__init__(verbose)
        self.degree_class = degree_class

    def get_metric(self, experiment, input, adj):
        raise NotImplementedError()

    def display(
        self, in_state, out_state, neighbor_state, ax=None, fill=None, **plot_kwargs
    ):
        if ax is None:
            ax = plt.gca()
        x = np.unique(np.sort(self.data["summaries"][:, neighbor_state + 1]))
        y = np.zeros(x.shape)
        err = np.zeros(x.shape)
        for i, xx in enumerate(x):
            index = (self.data["summaries"][:, neighbor_state + 1] == xx) * (
                self.data["summaries"][:, 0] == in_state
            )
            p = self.data["ltp"][index]
            y[i] = np.mean(self.data["ltp"][index, out_state], axis=0)
            err[i] = np.sqrt(
                (np.var(self.data["ltp"][index, out_state], axis=0)) / index.shape[0]
            )
        if fill is not None:
            ax.fill_between(x, y - err, y + err, color=fill, alpha=0.3)
        ax.plot(x, y, **plot_kwargs)
        return ax

    def compute(self, experiment):

        N = max(self.degree_class) + 1
        state_label = experiment.dynamics_model.state_label
        d = len(state_label)
        summaries = {}

        if self.verbose:
            num_iter = int(
                d * np.sum([binom(k + d - 1, d - 1) for k in self.degree_class])
            )
            p_bar = tqdm.tqdm(range(num_iter), "Computing " + self.__class__.__name__)

        for k in self.degree_class:
            adj = np.zeros((N, N))
            adj[1 : k + 1, :] = 1
            adj[:, 1 : k + 1] = 1
            for s in all_combinations(k, len(state_label)):
                neighbors_states = np.concatenate(
                    [i * np.ones(l) for i, l in enumerate(s)]
                )
                input = np.zeros(max(self.degree_class) + 1)
                input[1 : k + 1] = neighbors_states
                for in_s, in_l in state_label.items():
                    input[0] = in_l
                    summaries[(in_l, *s)] = self.get_metric(experiment, input, adj)
                    if self.verbose:
                        p_bar.update()

        if self.verbose:
            p_bar.close()

        self.data["summaries"] = np.array([s for s in summaries])
        self.data["ltp"] = np.array(
            [summaries[tuple(s)] for s in self.data["summaries"]]
        )


class DynamicsLTPGenMetrics(LTPGeneralizationMetrics):
    def __init__(self, verbose=1):
        super(DynamicsLTPGenMetrics, self).__init__(verbose)

    def get_metric(self, experiment, input, adj):
        return experiment.dynamics_model.predict(input, adj)[0]


class ModelLTPGenMetrics(LTPGeneralizationMetrics):
    def __init__(self, verbose=1):
        super(ModelLTPGenMetrics, self).__init__(verbose)

    def get_metric(self, experiment, input, adj):
        return experiment.model.predict(input, adj)[0]
