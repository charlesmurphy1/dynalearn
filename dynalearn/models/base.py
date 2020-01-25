import tensorflow as tf
import torch as pt
import numpy as np
import networkx as nx
from abc import ABC, abstractmethod


class GNNModel(ABC):
    def __init__(self,):
        self._model = None
        self._num_nodes = None
        self._adj = None
        self._graph = None

    @property
    def model(self):
        if self._model is None:
            self._model = self._prepare_model()
        return self._model

    @abstractmethod
    def _prepare_model(self):
        raise NotImplementedError("_prepare_model must be implemented.")

    @staticmethod
    @abstractmethod
    def loss_fct(y_true, y_pred):
        raise NotImplementedError("loss_fct must be implemented.")

    @property
    def num_nodes(self):
        if self._num_nodes is None:
            raise ValueError("Number of nodes has not been set.")
        else:
            return self._num_nodes

    @num_nodes.setter
    def num_nodes(self, num_nodes):
        self._num_nodes = num_nodes
        if self._model is None:
            self._model = self._prepare_model()
        else:
            weights = self._model.get_weights()
            self._model = self._prepare_model()
            self._model.set_weights(weights)

    @property
    def num_features(self):
        return self._num_features

    @num_features.setter
    def num_features(self, num_features):
        raise ValueError("num_features cannot be changed.")

    @property
    def graph(self):
        if self._graph is None:
            raise ValueError("No graph has been parsed to the dynamics.")
        else:
            return self._graph

    @graph.setter
    def graph(self, graph):
        self._graph = graph
        self._adj = nx.to_numpy_array(graph)
        if self.num_nodes != self._adj.shape[0]:
            self.num_nodes = self._adj.shape[0]

    @property
    def adj(self):
        if self._adj is None:
            raise ValueError("No graph has been parsed to the dynamics.")
        else:
            return self._adj

    @adj.setter
    def adj(self, adj):
        self._graph = nx.from_numpy_array(adj)
        self._adj = adj
        if self.num_nodes != self._adj.shape[0]:
            self.num_nodes = self._adj.shape[0]

    def predict(self, inputs):
        return self.model.predict([inputs, self.adj], batch_size=self.adj.shape[0])

    def sample(self, inputs):
        p = self.predict(inputs)
        p = pt.tensor(p)
        # if pt.cuda.is_available():
        #     p = p.cuda()
        dist = pt.distributions.Categorical(p)
        return np.array(dist.sample())
