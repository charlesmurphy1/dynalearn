import networkx as nx
import numpy as np

from abc import abstractmethod
from random import sample
from .metrics import Metrics


class PredictionMetrics(Metrics):
    def __init__(self, config, verbose=0):
        Metrics.__init__(self, config, verbose)
        self.max_num_points = config.pred_max_num_points
        self.model = None
        self.names = [
            "true",
            "pred",
            "degree",
            "train_true",
            "train_pred",
            "train_degree",
        ]

    def get_true(self, g_index, s_index):
        x = self.dataset._data["inputs"][g_index].data[s_index]
        return self.dynamics.predict(x)

    def get_pred(self, g_index, s_index):
        x = self.dataset.data["inputs"][g_index].data[s_index]
        return self.model.predict(x)

    def get_degrees(self):
        g = self.model.network
        if isinstance(g, dict):
            g = self.model.network["all"]
        return np.array(list(dict(g.degree()).values()))

    def get_network_true(self, index):
        self.dynamics.network = self.dataset._data["networks"][index].data

    def get_network_pred(self, index):
        self.model.network = self.dataset.data["networks"][index].data

    def initialize(self, experiment):
        self.model = experiment.model
        self.dynamics = experiment.dynamics
        self.dataset = experiment.dataset
        self.num_states = self.model.num_states
        self._get_points_()
        self.all_nodes = self._nodes_(experiment.dataset, all=True)
        self.get_data["true"] = lambda pb: self._pred_(
            self.get_true, self.get_network_true, self.all_nodes, pb=pb
        )
        self.get_data["pred"] = lambda pb: self._pred_(
            self.get_pred, self.get_network_pred, self.all_nodes, pb=pb
        )
        self.get_data["degree"] = lambda pb: self._degree_(self.all_nodes, pb=pb)

        train_nodes = self._nodes_(experiment.dataset)
        self.get_data["train_true"] = lambda pb: self._pred_(
            self.get_true, self.get_network_true, train_nodes, pb=pb
        )
        self.get_data["train_pred"] = lambda pb: self._pred_(
            self.get_pred, self.get_network_pred, train_nodes, pb=pb
        )
        self.get_data["train_degree"] = lambda pb: self._degree_(train_nodes, pb=pb)
        update_factor = 6
        if experiment.val_dataset is not None:
            val_nodes = self._nodes_(experiment.val_dataset)
            self.get_data["val_true"] = lambda pb: self._pred_(
                self.get_true, self.get_network_true, val_nodes, pb=pb
            )
            self.get_data["val_pred"] = lambda pb: self._pred_(
                self.get_pred, self.get_network_pred, val_nodes, pb=pb
            )
            self.get_data["val_degree"] = lambda pb: self._degree_(val_nodes, pb=pb)
            self.names.extend(["val_true", "val_true", "val_degree"])
            update_factor += 3

        if experiment.test_dataset is not None:
            test_nodes = self._nodes_(experiment.test_dataset)
            self.get_data["test_true"] = lambda pb: self._pred_(
                self.get_true, self.get_network_true, test_nodes, pb=pb
            )
            self.get_data["test_pred"] = lambda pb: self._pred_(
                self.get_pred, self.get_network_pred, test_nodes, pb=pb
            )
            self.get_data["test_degree"] = lambda pb: self._degree_(test_nodes, pb=pb)
            self.names.extend(["test_true", "test_true", "test_degree"])
            update_factor += 3
        self.num_updates *= update_factor

    def _get_points_(self):
        self.points = {}
        self.size = 0
        self.num_updates = 0
        for k, g in enumerate(self.dataset.networks.data_list):
            n = g.data.number_of_nodes()
            if (
                self.dataset.data["inputs"][k].size > self.max_num_points // n
                and self.max_num_points != -1
            ):
                self.points[k] = sample(
                    range(self.dataset.data["inputs"][k].size),
                    int(self.max_num_points // n),
                )
                self.size += self.max_num_points
            else:
                self.points[k] = range(self.dataset.data["inputs"][k].size)
                self.size += self.dataset.data["inputs"][k].size * n
            self.num_updates += len(self.points[k])

    def _nodes_(self, dataset, all=False):
        weights = dataset.weights
        nodes = {}

        for g_index in range(dataset.networks.size):
            nodes[g_index] = {}
            for s_index in range(dataset.inputs[g_index].size):
                if all:
                    nodes[g_index][s_index] = np.arange(
                        dataset.weights[g_index].data[s_index].shape[0]
                    )
                else:
                    nodes[g_index][s_index] = np.where(
                        dataset.weights[g_index].data[s_index] > 0
                    )[0]
        return nodes

    def _pred_(self, pred_getter, net_getter, nodes, pb=None):
        pred_array = np.zeros([int(self.size), int(self.num_states)])
        i = 0
        for k in range(self.dataset.data["networks"].size):
            indices = set(range(self.dataset.data["inputs"][k].size))
            net_getter(k)
            for t in self.points[k]:
                pred = pred_getter(k, t)[nodes[k][t], :]
                if i + pred.shape[0] <= pred_array.shape[0]:
                    pred_array[i : i + pred.shape[0]] = pred
                    i = i + pred.shape[0]
                else:
                    index = sample(range(pred.shape[0]), pred_array[i:].shape[0])
                    pred_array[i:] = pred[index]
                    break
                if pb is not None:
                    pb.update()

        return pred_array

    def _degree_(self, nodes, pb=None):
        degree_array = np.zeros(int(self.size))
        i = 0
        for k in range(self.dataset.data["networks"].size):
            indices = set(range(self.dataset.data["inputs"][k].size))
            self.get_network_pred(k)
            degree = self.get_degrees()
            for t in self.points[k]:
                deg = degree[nodes[k][t]]
                if i + deg.shape[0] <= degree_array.shape[0]:
                    degree_array[i : i + deg.shape[0]] = deg
                    i = i + deg.shape[0]
                else:
                    index = sample(range(deg.shape[0]), degree_array[i:].shape[0])
                    degree_array[i:] = deg[index]
                    break
                if pb is not None:
                    pb.update()

        return degree_array


# class TruePredictionMetrics(PredictionMetrics):
#     def get_model(self, experiment):
#         return experiment.dynamics
#
#     def get_prediction(self, g_index, s_index):
#         x = self.dataset._data["inputs"][g_index].data[s_index]
#         return self.model.predict(x)
#
#     def get_network(self, index):
#         return self.dataset._data["networks"][index].data
#
#
# class GNNPredictionMetrics(PredictionMetrics):
#     def get_model(self, experiment):
#         return experiment.model
#
#     def get_prediction(self, g_index, s_index):
#         x = self.dataset.data["inputs"][g_index].data[s_index]
#         return self.model.predict(x)
