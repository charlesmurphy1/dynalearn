import networkx as nx
import numpy as np

from abc import abstractmethod
from .base import Metrics
from dynalearn.utilities import all_combinations
from itertools import product


class LTPMetrics(Metrics):
    def __init__(self, config, verbose=0):
        Metrics.__init__(self, config, verbose)

        self.max_num_sample = config.max_num_sample
        self.max_num_points = config.max_num_points

        self.model = None
        self.networks = {}
        self.inputs = {}
        self.targets = {}

        self.all_nodes = {}
        self.summaries = set()

        self.names = [
            "summaries",
            "ltp",
            "train_ltp",
        ]

    @abstractmethod
    def get_model(self, experiment):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, x, y):
        raise NotImplementedError()

    def initialize(self, experiment):
        self.model = self.get_model(experiment)
        self.networks = experiment.dataset.networks
        self.inputs = experiment.dataset.inputs
        self.targets = experiment.dataset.targets

        self.num_points = {}
        self.num_updates = 0
        for k, g in self.networks.items():
            if (
                self.max_num_points < self.inputs[k].shape[0]
                and self.max_num_points > 1
            ):
                self.num_points[k] = self.max_num_points
            else:
                self.num_points[k] = self.inputs[k].shape[0]
                self.num_updates += self.num_points[k]
        self.get_data["summaries"] = self._get_summaries_

        self.all_nodes = self._get_nodes_(experiment.dataset, all=True)

        self.get_data["ltp"] = lambda pb: self._get_ltp_(self.all_nodes, pb=pb)

        train_nodes = self._get_nodes_(experiment.dataset)
        self.get_data["train_ltp"] = lambda pb: self._get_ltp_(train_nodes, pb=pb)
        factor = 2

        if experiment.val_dataset is not None:
            val_nodes = self._get_nodes_(experiment.val_dataset)
            self.get_data["val_ltp"] = lambda pb: self._get_ltp_(val_nodes, pb=pb)
            self.names.append("val_ltp")
            factor += 1
        if experiment.test_dataset is not None:
            test_nodes = self._get_nodes_(experiment.test_dataset)
            self.names.append("test_ltp")
            factor += 1
        self.num_updates *= factor

    def _get_summaries_(self, pb=None):
        for k, g in self.networks.items():
            self.model.network = g
            adj = nx.to_numpy_array(g)
            for t in range(self.num_points[k]):
                x = self.inputs[k][t]
                l = np.array(
                    [np.matmul(adj, x == i) for i in range(self.model.num_states)]
                ).T
                for i in self.all_nodes[k][t]:
                    s = (x[i], *list(l[i]))
                    if s not in self.summaries:
                        self.summaries.add(s)
        return np.array(list(self.summaries))

    def _get_ltp_(self, nodes, pb=None):
        ltp = {}
        counter = {}

        for k, g in self.networks.items():
            self.model.network = g
            adj = nx.to_numpy_array(g)
            for t in range(self.num_points[k]):
                x = self.inputs[k][t]
                l = np.array(
                    [np.matmul(adj, x == i) for i in range(self.model.num_states)]
                ).T

                y = self.targets[k][t]
                pred = self.predict(x, y)
                for i in nodes[k][t]:
                    s = (x[i], *list(l[i]))
                    if s in ltp:
                        if counter[s] == self.max_num_sample:
                            continue
                        ltp[s][counter[s]] = pred[i]
                        counter[s] += 1
                    else:
                        ltp[s] = (
                            np.ones((self.max_num_sample, self.model.num_states))
                            * np.nan
                        )
                        ltp[s][0] = pred[i]
                        counter[s] = 1
                if pb is not None:
                    pb.update()
        ltp_array = np.ones((len(self.summaries), self.model.num_states)) * np.nan
        for i, s in enumerate(self.summaries):
            index = np.nansum(ltp[s], axis=-1) > 0
            if np.sum(index) > 0:
                ltp_array[i] = np.mean(ltp[s][index], axis=0).squeeze()

        return ltp_array

    def _get_nodes_(self, dataset, all=True):
        weights = dataset.weights
        nodes = {}

        for g_index in range(len(dataset.networks)):
            nodes[g_index] = {}
            for s_index in range(dataset.inputs[g_index].shape[0]):
                if all:
                    nodes[g_index][s_index] = np.arange(
                        dataset.weights[g_index][s_index].shape[0]
                    )
                else:
                    nodes[g_index][s_index] = np.where(
                        dataset.weights[g_index][s_index] > 0
                    )[0]
        return nodes

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
        if reduce is "mean":
            op = np.nanmean
            if err_reduce is "std":
                err_op = lambda xx: (
                    np.nanmean(xx) - np.nanstd(xx),
                    np.nanmean(xx) + np.nanstd(xx),
                )
            elif err_reduce is "percentile":
                err_op = lambda xx: (np.nanpercentile(xx, 16), np.nanpercentile(xx, 84))
            else:
                raise ValueError(
                    "Invalid error reduction, valid options are ['std', 'percentile']"
                )
        elif reduce is "sum":
            op = np.nansum
            err_op = lambda x: np.nan
        else:
            raise ValueError(
                "Invalid error reduction, valid options are ['mean', 'sum']"
            )

        if axis == -1:
            all_summ = summaries[:, 1:].sum(-1)
        else:
            all_summ = summaries[:, axis + 1]
        agg_summ = np.unique(np.sort(all_summ))
        agg_ltp = np.zeros(agg_summ.shape)
        agg_ltp_low = np.zeros(agg_summ.shape)
        agg_ltp_high = np.zeros(agg_summ.shape)
        for i, x in enumerate(agg_summ):

            if in_state is None:
                index = all_summ == x
            else:
                index = (all_summ == x) * (summaries[:, 0] == in_state)

            if out_state is None or len(data.shape) == 1:
                y = data[index]
            else:
                y = data[index, out_state]

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

    def predict(self, x, y):
        return self.model.predict(x)


class GNNLTPMetrics(LTPMetrics):
    def __init__(self, config, verbose=0):
        LTPMetrics.__init__(self, config, verbose)

    def get_model(self, experiment):
        return experiment.model

    def predict(self, x, y):
        return self.model.predict(x)


class MLELTPMetrics(LTPMetrics):
    def __init__(self, config, verbose=0):
        LTPMetrics.__init__(self, config, verbose)
        if "mle_num_points" in config.__dict__:
            self.num_points = config.mle_num_points

    def get_model(self, experiment):
        return experiment.dynamics

    def predict(self, x, y):
        one_hot_target = np.zeros((y.shape[0], self.model.num_states), dtype="int")
        one_hot_target[np.arange(y.shape[0]), y.astype("int")] = 1
        return one_hot_target


class UniformLTPMetrics(LTPMetrics):
    def __init__(self, config, verbose=0):
        LTPMetrics.__init__(self, config, verbose)

    def get_model(self, experiment):
        return experiment.dynamics

    def predict(self, x, y):
        return np.ones((x.shape[0], self.model.num_states)) / self.model.num_states


class StarLTPMetrics(LTPMetrics):
    def __init__(self, config, verbose=0):
        LTPMetrics.__init__(self, config, verbose)
        self.degree_class = config.degree_class
        self.names = [
            "summaries",
            "ltp",
        ]

    def initialize(self, experiment):
        self.model = self.get_model(experiment)
        self.networks = {}
        self.inputs = {}
        self.targets = {}

        self.num_points = {}
        self.all_nodes = {}
        self.num_updates = 0
        num_nodes = np.max(self.degree_class) + 1
        for k in self.degree_class:
            self.networks[k] = nx.empty_graph(num_nodes)
            self.networks[k].add_edges_from(nx.star_graph(k).edges())
            self.inputs[k] = {}
            self.targets[k] = {}
            all_s = range(self.model.num_states)
            all_ns = all_combinations(k, self.model.num_states)
            self.all_nodes[k] = {}
            self.num_points[k] = 0

            for i, (s, ns) in enumerate(product(all_s, all_ns)):
                self.num_points[k] += 1
                self.num_updates += 1
                self.inputs[k][i] = np.zeros(num_nodes)
                self.targets[k][i] = np.zeros(num_nodes)
                self.inputs[k][i][0] = s
                self.inputs[k][i][1 : k + 1] = np.concatenate(
                    [j * np.ones(l) for j, l in enumerate(ns)]
                )
                self.all_nodes[k][i] = list(range(num_nodes))

        self.get_data["summaries"] = self._get_summaries_
        self.get_data["ltp"] = lambda pb: self._get_ltp_(self.all_nodes, pb=pb)


class TrueStarLTPMetrics(TrueLTPMetrics, StarLTPMetrics):
    def __init__(self, config, verbose=0):
        TrueLTPMetrics.__init__(self, config, verbose)
        StarLTPMetrics.__init__(self, config, verbose)


class GNNStarLTPMetrics(GNNLTPMetrics, StarLTPMetrics):
    def __init__(self, config, verbose=0):
        GNNLTPMetrics.__init__(self, config, verbose)
        StarLTPMetrics.__init__(self, config, verbose)


class UniformStarLTPMetrics(UniformLTPMetrics, StarLTPMetrics):
    def __init__(self, config, verbose=0):
        UniformLTPMetrics.__init__(self, config, verbose)
        StarLTPMetrics.__init__(self, config, verbose)
