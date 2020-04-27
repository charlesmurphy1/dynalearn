import networkx as nx
import numpy as np
import tqdm

from abc import abstractmethod
from .metrics import Metrics
from itertools import product
from scipy.special import binom


class StatisticsMetrics(Metrics):
    def __init__(self, config, verbose=0):
        Metrics.__init__(self, config, verbose)
        self.max_num_sample = config.max_num_sample
        self.max_num_points = config.max_num_points

        self.networks = {}
        self.inputs = {}
        self.targets = {}

        self.all_nodes = {}

        self.names = [
            "summaries",
            "all_stats",
            "all_entropy",
            "all_ess",
            "train_stats",
            "train_entropy",
            "train_ess",
        ]

        self.summaries = set()

    def initialize(self, experiment):
        self.networks = experiment.dataset.networks
        self.inputs = experiment.dataset.inputs
        self.targets = experiment.dataset.targets
        self.num_states = experiment.dynamics.num_states

        self.num_points = {}
        self.num_updates = 0
        for k, g in self.networks.items():
            self.num_points[k] = self.inputs[k].shape[0]
            self.num_updates += self.num_points[k]
        self.get_data["summaries"] = self._get_summaries_

        self.all_nodes = self._get_nodes_(experiment.dataset, all=True)
        self.get_data["all_stats"] = lambda pb: self._get_stats_(self.all_nodes, pb=pb)
        self.get_data["all_entropy"] = lambda pb: self._get_entropy_("all", pb=pb)
        self.get_data["all_ess"] = lambda pb: self._get_ess_("all", pb=pb)

        train_nodes = self._get_nodes_(experiment.dataset, all=False)
        self.get_data["train_stats"] = lambda pb: self._get_stats_(train_nodes, pb=pb)
        self.get_data["train_entropy"] = lambda pb: self._get_entropy_("train", pb=pb)
        self.get_data["train_ess"] = lambda pb: self._get_ess_("train", pb=pb)
        factor = 2

        if experiment.val_dataset is not None:
            val_nodes = self._get_nodes_(experiment.val_dataset, all=False)
            self.names.extend(["val_stats", "val_entropy", "val_ess"])
            self.get_data["val_stats"] = lambda pb: self._get_stats_(val_nodes, pb=pb)
            self.get_data["val_entropy"] = lambda pb: self._get_entropy_("val", pb=pb)
            self.get_data["val_ess"] = lambda pb: self._get_ess_("val", pb=pb)
            factor += 1

        if experiment.test_dataset is not None:
            test_nodes = self._get_nodes_(experiment.test_dataset, all=False)
            self.names.extend(["test_stats", "test_entropy", "test_ess"])
            self.get_data["test_stats"] = lambda pb: self._get_stats_(test_nodes, pb=pb)
            self.get_data["test_entropy"] = lambda pb: self._get_entropy_("test", pb=pb)
            self.get_data["test_ess"] = lambda pb: self._get_ess_("test", pb=pb)
            factor += 1

        self.num_updates *= factor

    def _get_summaries_(self, pb=None):
        for k, g in self.networks.items():
            adj = nx.to_numpy_array(g)
            for t in range(self.num_points[k]):
                x = self.inputs[k][t]
                l = np.array([np.matmul(adj, x == i) for i in range(self.num_states)]).T
                for i in self.all_nodes[k][t]:
                    s = (x[i], *list(l[i]))
                    if s not in self.summaries:
                        self.summaries.add(s)
        return np.array(list(self.summaries))

    def _get_stats_(self, nodes, pb=None):
        stats = {}

        for k, g in self.networks.items():
            adj = nx.to_numpy_array(g)
            for t in range(self.num_points[k]):
                x = self.inputs[k][t]
                l = np.array([np.matmul(adj, x == i) for i in range(self.num_states)]).T
                y = self.targets[k][t]
                for i in nodes[k][t]:
                    s = (x[i], *list(l[i]))
                    if s in stats:
                        stats[s] += 1
                    else:
                        stats[s] = 1
                if pb is not None:
                    pb.update()
        stats_array = np.zeros(len(self.summaries))
        for i, s in enumerate(self.summaries):
            if s in stats:
                stats_array[i] = stats[s]

        return stats_array

    def _get_entropy_(self, dataset, pb=None):
        x = self.data[dataset + "_stats"]

        p = x / x.sum()
        p = p[p > 0]
        entropy = -np.sum(p * np.log(p)) / self.max_entropy()
        return entropy

    def _get_ess_(self, dataset, pb=None):
        x = self.data[dataset + "_stats"]
        ess = np.sum(x) ** 2 / np.sum(x ** 2)
        return ess

    def max_entropy(self):
        summaries = np.array(list(self.summaries))
        degrees = np.unique(np.sort(np.sum(summaries[:, 1:], axis=-1)))
        degrees = degrees[degrees > 0]
        num_states = summaries[0, 1:].shape[0]
        ans = 0
        for k in degrees:
            ans += num_states * binom(num_states + k - 1, k)
        max_entropy = np.log(ans)
        return max_entropy

    def _get_nodes_(self, dataset, all=False):
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
