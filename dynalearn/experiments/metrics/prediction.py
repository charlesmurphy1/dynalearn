import networkx as nx
import numpy as np

from abc import abstractmethod
from .metrics import Metrics


class PredictionMetrics(Metrics):
    def __init__(self, config, verbose=0):
        Metrics.__init__(self, config, verbose)

        self.max_num_points = config.max_num_points
        self.model = None
        self.names = ["pred", "degree", "train_pred", "train_degree"]

    @abstractmethod
    def get_model(self, experiment):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, real_x, obs_x, real_y, obs_y):
        raise NotImplementedError()

    def _set_network_(self, real_g, obs_g):
        return obs_g

    def initialize(self, experiment):
        self.model = self.get_model(experiment)
        self.dataset = experiment.dataset
        self.num_states = experiment.model.num_states

        self.num_points = {}
        self.num_updates = 0

        for i in range(self.dataset.networks.size):
            g = self.dataset.networks[i].data
            n = g.number_of_nodes()
            if (
                self.max_num_points < self.dataset.inputs[i].size * n
                and self.max_num_points > 1
            ):
                self.num_points[i] = self.max_num_points
            else:
                self.num_points[i] = self.dataset.inputs[i].size * n
            self.num_updates += self.dataset.inputs[i].size

        self.all_nodes = self._get_nodes_(experiment.dataset, all=True)
        self.get_data["pred"] = lambda pb: self._get_pred_(self.all_nodes, pb=pb)
        self.get_data["degree"] = lambda pb: self._get_degree_(self.all_nodes)

        train_nodes = self._get_nodes_(experiment.dataset)
        self.get_data["train_pred"] = lambda pb: self._get_pred_(train_nodes, pb=pb)
        self.get_data["train_degree"] = lambda pb: self._get_degree_(train_nodes)
        update_factor = 2
        if experiment.val_dataset is not None:
            val_nodes = self._get_nodes_(experiment.val_dataset)
            self.get_data["val_pred"] = lambda pb: self._get_pred_(val_nodes, pb=pb)
            self.get_data["val_degree"] = lambda pb: self._get_degree_(val_nodes)
            self.names.extend(["val_pred", "val_degree"])
            update_factor += 1

        if experiment.test_dataset is not None:
            test_nodes = self._get_nodes_(experiment.test_dataset)
            self.get_data["test_pred"] = lambda pb: self._get_pred_(test_nodes, pb=pb)
            self.get_data["test_degree"] = lambda pb: self._get_degree_(test_nodes)
            self.names.extend(["test_pred", "test_degree"])
            update_factor += 1
        self.num_updates *= update_factor

    def _get_nodes_(self, dataset, all=False):
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

    def _get_pred_(self, nodes, pb=None):
        pred_array = np.zeros([sum(self.num_points.values()), self.num_states])
        i = 0
        for k in range(self.dataset.networks.size):
            obs_g = self.dataset.data["networks"][k].data
            real_g = self.dataset._data["networks"][k].data
            self.model.network = self._set_network_(real_g, obs_g)
            for t in range(self.dataset.data["inputs"][k].size):
                real_x = self.dataset._data["inputs"][k].data[t]
                obs_x = self.dataset.data["inputs"][k].data[t]
                real_y = self.dataset._data["targets"][k].data[t]
                obs_y = self.dataset.targets[k].data[t]
                pred = self.predict(real_x, obs_x, real_y, obs_y)[nodes[k][t], :]
                if i + pred.shape[0] <= pred_array.shape[0]:
                    pred_array[i : i + pred.shape[0]] = pred
                    i = i + pred.shape[0]
                else:
                    index = np.random.choice(
                        range(pred.shape[0]),
                        size=pred_array[i:].shape[0],
                        replace=False,
                    )
                    pred_array[i:] = pred[index]
                    break
                if pb is not None:
                    pb.update()

        return pred_array

    def _get_degree_(self, nodes):
        degree_array = np.zeros(sum(self.num_points.values()))
        i = 0
        for k in range(self.dataset.networks.size):
            g = self.dataset.data["networks"][k].data
            degree_seq = np.array(list(dict(g.degree()).values()))
            for t in range(self.dataset.data["inputs"][k].size):
                degree = degree_seq[nodes[k][t]]
            if i + degree.shape[0] <= degree.shape[0]:
                degree_array[i : i + degree.shape[0]] = degree
                i = i + degree.shape[0]
            else:
                index = np.random.choice(
                    range(degree.shape[0]),
                    size=degree_array[i:].shape[0],
                    replace=False,
                )
                degree_array[i:] = degree[index]
                break
        return degree_array


class TruePredictionMetrics(PredictionMetrics):
    def __init__(self, config, verbose=0):
        PredictionMetrics.__init__(self, config, verbose)

    def get_model(self, experiment):
        return experiment.dynamics

    def predict(self, real_x, obs_x, real_y, obs_y):
        return self.model.predict(real_x)

    def _set_network_(self, real_g, obs_g):
        return real_g


class GNNPredictionMetrics(PredictionMetrics):
    def __init__(self, config, verbose=0):
        PredictionMetrics.__init__(self, config, verbose)

    def get_model(self, experiment):
        return experiment.model

    def predict(self, real_x, obs_x, real_y, obs_y):
        return self.model.predict(obs_x)
