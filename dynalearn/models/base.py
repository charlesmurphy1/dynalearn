import tensorflow as tf
import torch as pt
import networkx as nx
import numpy as np
from abc import ABC, abstractmethod


class GNNModel(ABC):
    def __init__(self,):
        self._model = None
        self._graph = None
        self._adj = None
        self._num_nodes = None
        self._degree = None

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
    def graph(self):
        if self._graph is None:
            raise ValueError("No graph has been parsed to the dynamics.")
        else:
            return self._graph

    @graph.setter
    def graph(self, graph):
        self._graph = graph
        self._adj = nx.to_numpy_array(graph)
        self.num_nodes = self._graph.number_of_nodes()
        self._degree = np.sum(self._adj, axis=1)

    @property
    def adj(self):
        if self._adj is None:
            raise ValueError("No graph has been parsed to the dynamics.")
        else:
            return self._adj

    @adj.setter
    def adj(self, adj):
        self._adj = adj
        self.graph = nx.from_numpy_array(adj)
        self._num_nodes = self._graph.number_of_nodes()
        self._degree = np.sum(self._adj, axis=1)

    @property
    def num_nodes(self):
        if self._num_nodes is None:
            raise ValueError("No graph has been parsed to the dynamics.")
        else:
            return self._num_nodes

    @num_nodes.setter
    def num_nodes(self, num_nodes):
        if num_nodes == self._num_nodes:
            return
        self._num_nodes = num_nodes
        if self._model is None:
            self._model = self._prepare_model()
        else:
            weights = self._model.get_weights()
            self._model = self._prepare_model()
            self._model.set_weights(weights)

    @property
    def degree(self):
        if self._degree is None:
            raise ValueError("No graph has been parsed to the dynamics.")
        else:
            return self._degree

    @property
    def num_states(self):
        return self._num_states

    @num_states.setter
    def num_states(self, num_states):
        self._num_states = num_states

    def predict(self, inputs):
        return self.model.predict([inputs, self.adj], steps=1)

    def sample(self, inputs):
        p = self.predict(inputs)
        dist = pt.distributions.Categorical(pt.tensor(p))
        return np.array(dist.sample())
