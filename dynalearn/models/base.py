import tensorflow as tf
import torch as pt
import numpy as np


class DynamicsPredictor:
    def __init__(self, num_nodes, num_states):
        self._model = None
        self._num_nodes = num_nodes
        self._num_states = num_states
        self.params = {}
        # print(tf.get_seed(None))

    @property
    def model(self):
        if self._model is None:
            self._model = self._prepare_model()
        return self._model

    def _prepare_model(self):
        raise NotImplemented("_prepare_model() must be implemented")

    @property
    def num_nodes(self):
        if self._num_nodes is None:
            raise ValueError("No graph has been given to the dynamics.")
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
    def num_states(self):
        if self._num_states is None:
            raise ValueError("No graph has been given to the dynamics.")
        else:
            return self._num_states

    @num_states.setter
    def num_states(self, num_states):
        raise ValueError("num_states cannot be changed.")

    def predict(self, inputs, adj):
        return self.model.predict([inputs, adj], steps=1)

    def update(self, inputs, adj):
        p = self.predict(inputs, adj)
        dist = pt.distributions.Categorical(pt.tensor(p))
        return np.array(dist.sample())
