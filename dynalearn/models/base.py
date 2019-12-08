import tensorflow as tf
import torch as pt
import numpy as np
from abc import ABC, abstractmethod


class GNNModel(ABC):
    def __init__(self,):
        self._model = None
        self._num_nodes = None

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

    def predict(self, inputs, adj):
        n = adj.shape[0]
        if n != self.num_nodes:
            self.num_nodes = n
        return self.model.predict([inputs, adj], steps=1)

    def sample(self, inputs, adj):
        n = adj.shape[0]
        if n != self.num_nodes:
            self.num_nodes = n
        p = self.predict(inputs, adj)
        dist = pt.distributions.Categorical(pt.tensor(p))
        return np.array(dist.sample())
