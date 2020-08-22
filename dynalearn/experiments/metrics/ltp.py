import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from abc import abstractmethod
from .metrics import Metrics
from dynalearn.dynamics import Epidemics
from dynalearn.utilities import all_combinations, from_nary
from itertools import product
from scipy.special import binom


class LTPMetrics(Metrics):
    def __init__(self, config, verbose=0):
        Metrics.__init__(self, config, verbose)

        self.max_num_sample = config.max_num_sample
        self.max_num_points = config.max_num_points

        self.model = None

        self.all_nodes = {}
        self.summaries = set()

        self.names = ["summaries", "ltp", "train_ltp"]

    def display(
        self,
        ax=None,
        data_name=None,
        in_states=[],
        out_states=[],
        colors=None,
        fill_color=None,
        fill_alpha=0.2,
        axis=None,
        **kwargs,
    ):
        if isinstance(in_states, int) or isinstance(in_states, tuple):
            in_states = [in_states]

        if isinstance(out_states, int) or isinstance(out_states, tuple):
            out_states = [out_states]

        ax = ax or plt.gca()
        data_name = data_name or "ltp"
        colors = colors or {(i, j): "k" for (i, j) in product(in_states, out_states)}
        fill_color = "k"

        for i, j in product(in_states, out_states):
            x, y, yl, yh = LTPMetrics.aggregate(
                self.data[data_name],
                self.data["summaries"],
                in_state=i,
                out_state=i,
                axis=axis,
                reduce="mean",
                err_reduce="percentile",
            )
            c = colors[(i, j)]
            ax.plot(x, y, color=c, **kwargs)
            ax.fill_between(x, yl, yh, color=fill_color, alpha=0.2)
        return ax

    @abstractmethod
    def get_model(self, experiment):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, real_x, obs_x, real_y, obs_y):
        raise NotImplementedError()

    def initialize(self, experiment):
        self.model = self.get_model(experiment)
        if not issubclass(Epidemics, self.model.__class__):
            raise ValueError(f"{self.model} is an invalid model for LTPMetrics.")
        self.dataset = experiment.dataset
        self.num_states = experiment.model.num_states
        if (
            experiment.model.window_size
            > experiment.train_details.threshold_window_size
        ):
            self.window_size = experiment.train_details.threshold_window_size
        else:
            self.window_size = experiment.model.window_size

        self.num_points = {}
        self.num_updates = 0

        for k, g in enumerate(self.dataset.networks.data):
            if (
                self.max_num_points < self.dataset.inputs[k].size
                and self.max_num_points > 1
            ):
                self.num_points[k] = self.max_num_points
            else:
                self.num_points[k] = self.dataset.inputs[k].size
                self.num_updates += self.num_points[k]

        self.get_data["summaries"] = self._get_summaries_
        self.all_nodes = self._get_nodes_(experiment.dataset, all=True)
        self.get_data["ltp"] = lambda pb: self._get_ltp_(self.all_nodes, pb=pb)

        train_nodes = self._get_nodes_(experiment.dataset)
        self.get_data["train_ltp"] = lambda pb: self._get_ltp_(train_nodes, pb=pb)
        update_factor = 2

        if experiment.val_dataset is not None:
            val_nodes = self._get_nodes_(experiment.val_dataset)
            self.get_data["val_ltp"] = lambda pb: self._get_ltp_(val_nodes, pb=pb)
            self.names.append("val_ltp")
            update_factor += 1

        if experiment.test_dataset is not None:
            test_nodes = self._get_nodes_(experiment.test_dataset)
            self.get_data["test_ltp"] = lambda pb: self._get_ltp_(test_nodes, pb=pb)
            self.names.append("test_ltp")
            update_factor += 1
        self.num_updates *= update_factor

    def _get_summaries_(self, pb=None):
        eff_num_states = self.num_states ** self.window_size

        for k in range(self.dataset.networks.size):
            g = self.dataset.networks.data[k]
            adj = nx.to_numpy_array(g)
            for t in range(self.num_points[k]):
                obs_x = self.dataset.data["inputs"][k][t]
                obs_x = from_nary(
                    obs_x[: self.window_size], axis=0, base=self.num_states
                )
                l = np.array(
                    [np.matmul(adj, obs_x == i) for i in range(eff_num_states)]
                ).T
                for i in self.all_nodes[k][t]:
                    s = (obs_x[i], *list(l[i]))
                    if s not in self.summaries:
                        self.summaries.add(s)
        return np.array(list(self.summaries))

    def _get_ltp_(self, nodes, pb=None):
        ltp = {}
        counter = {}
        eff_num_states = self.num_states ** self.window_size

        for k in range(self.dataset.networks.size):
            real_g = self.dataset._data["networks"].data[k]
            obs_g = self.dataset.data["networks"].data[k]
            self.model.network = self._set_network_(real_g, obs_g)
            adj = nx.to_numpy_array(obs_g)
            for t in range(self.num_points[k]):
                real_x = self.dataset._data["inputs"][k].data[t]
                obs_x = self.dataset.data["inputs"][k].data[t]
                real_y = self.dataset._data["targets"][k].data[t]
                obs_y = self.dataset.targets[k][t]
                pred = self.predict(real_x, obs_x, real_y, obs_y)

                bin_x = (
                    from_nary(
                        obs_x[:, -self.window_size :], axis=0, base=self.num_states
                    )
                    * 1
                )
                l = np.array(
                    [np.matmul(adj, bin_x == i) for i in range(eff_num_states)]
                ).T
                for i in nodes[k][t]:
                    s = (bin_x[i], *list(l[i]))
                    if s in ltp:
                        if counter[s] == self.max_num_sample:
                            continue
                        ltp[s][counter[s]] = pred[i]
                        counter[s] += 1
                    else:
                        ltp[s] = (
                            np.ones((self.max_num_sample, self.num_states)) * np.nan
                        )
                        ltp[s][0] = pred[i]
                        counter[s] = 1
                if pb is not None:
                    pb.update()
        ltp_array = np.ones((len(self.summaries), self.num_states)) * np.nan
        for i, s in enumerate(self.summaries):
            index = np.nansum(ltp[s], axis=-1) > 0
            if np.sum(index) > 0:
                ltp_array[i] = np.mean(ltp[s][index], axis=0).squeeze()

        return ltp_array

    def _get_nodes_(self, dataset, all=False):
        weights = dataset.weights
        nodes = {}

        for g_index in range(dataset.networks.size):
            nodes[g_index] = {}
            for s_index in range(dataset.inputs[g_index].size):
                if all:
                    nodes[g_index][s_index] = np.arange(
                        dataset.weights[g_index][s_index].shape[0]
                    )
                else:
                    nodes[g_index][s_index] = np.where(
                        dataset.weights[g_index][s_index] > 0
                    )[0]
        return nodes

    def _set_network_(self, real_g, obs_g):
        return obs_g

    @staticmethod
    def aggregate(
        data,
        summaries,
        in_state=None,
        out_state=None,
        axis=-1,
        reduce="mean",
        err_reduce="std",
    ):
        if reduce == "mean":
            op = np.nanmean
            if err_reduce == "std":
                err_op = lambda xx: (
                    np.nanmean(xx) - np.nanstd(xx),
                    np.nanmean(xx) + np.nanstd(xx),
                )
            elif err_reduce == "percentile":
                err_op = lambda xx: (np.nanpercentile(xx, 16), np.nanpercentile(xx, 84))
            else:
                raise ValueError(
                    f"{err_reduce} is an invalid reduction, valid options are ['std', 'percentile']"
                )
        elif reduce == "sum":
            op = np.nansum
            err_op = lambda x: np.nan
        else:
            raise ValueError(
                "Invalid error reduction, valid options are ['mean', 'sum']"
            )

        if axis == -1:
            all_summ = summaries[:, 1:].sum(-1)
        elif isinstance(axis, int) or isinstance(axis, float):
            all_summ = summaries[:, int(axis + 1)]
        elif isinstance(axis, tuple):
            all_summ = np.zeros(summaries.shape[0])

            for i, a in enumerate(axis):
                all_summ += summaries[:, a + 1]
        agg_summ = np.unique(np.sort(all_summ))

        agg_ltp = np.zeros(agg_summ.shape)
        agg_ltp_low = np.zeros(agg_summ.shape)
        agg_ltp_high = np.zeros(agg_summ.shape)
        for i, x in enumerate(agg_summ):
            cond1 = all_summ == x

            if in_state is None:
                cond2 = cond1
            elif isinstance(in_state, int) or isinstance(in_state, float):
                cond2 = summaries[:, 0] == in_state
            elif isinstance(in_state, tuple):
                cond2 = np.logical_or(*[summaries[:, 0] == s for s in in_state])
            index = np.logical_and(cond1, cond2)

            if out_state is None or len(data.shape) == 1:
                y = data[index]
            else:
                if isinstance(out_state, int):
                    y = data[index, out_state]
                elif isinstance(out_state, tuple):
                    y = np.array([data[index, o] for o in out_state])
                    y = y.sum(0)

            if len(y) > 0:
                agg_ltp[i] = op(y)
                agg_ltp_low[i], agg_ltp_high[i] = err_op(y)
            else:
                agg_ltp[i] = np.nan
        agg_summ = agg_summ[~np.isnan(agg_ltp)]
        agg_ltp_low = agg_ltp_low[~np.isnan(agg_ltp)]
        agg_ltp_high = agg_ltp_high[~np.isnan(agg_ltp)]
        agg_ltp = agg_ltp[~np.isnan(agg_ltp)]
        return agg_summ, agg_ltp, agg_ltp_low, agg_ltp_high

    @staticmethod
    def compare(data1, data2, summaries, func=None):
        comparison = np.zeros(summaries.shape[0])
        for i, s in enumerate(summaries):
            comparison[i] = func(data1[i], data2[i])
        return comparison


class TrueLTPMetrics(LTPMetrics):
    def __init__(self, config, verbose=0):
        LTPMetrics.__init__(self, config, verbose)

    def get_model(self, experiment):
        return experiment.dynamics

    def predict(self, real_x, obs_x, real_y, obs_y):
        return self.model.predict(real_x)

    def _set_network_(self, real_g, obs_g):
        return real_g


class GNNLTPMetrics(LTPMetrics):
    def __init__(self, config, verbose=0):
        LTPMetrics.__init__(self, config, verbose)

    def get_model(self, experiment):
        return experiment.model

    def predict(self, real_x, obs_x, real_y, obs_y):
        return self.model.predict(obs_x)


class MLELTPMetrics(LTPMetrics):
    def __init__(self, config, verbose=0):
        LTPMetrics.__init__(self, config, verbose)
        if "mle_num_points" in config.__dict__:
            self.num_points = config.mle_num_points

    def get_model(self, experiment):
        return experiment.dynamics

    def predict(self, real_x, obs_x, real_y, obs_y):
        return obs_y


class UniformLTPMetrics(LTPMetrics):
    def __init__(self, config, verbose=0):
        LTPMetrics.__init__(self, config, verbose)

    def get_model(self, experiment):
        return experiment.dynamics

    def predict(self, real_x, obs_x, real_y, obs_y):
        return np.ones((obs_x.shape[0], self.num_states)) / self.num_states
