import torch
import torch.nn as nn

from .gnn import (
    GraphNeuralNetwork,
    WeightedGraphNeuralNetwork,
    MultiplexGraphNeuralNetwork,
    WeightedMultiplexGraphNeuralNetwork,
)
from dynalearn.config import Config
from dynalearn.nn.loss import weighted_cross_entropy


class DeterministicEpidemicsGNN(GraphNeuralNetwork):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        self.num_states = config.num_states
        GraphNeuralNetwork.__init__(
            self,
            config.num_states,
            config.num_states,
            window_size=config.window_size,
            nodeattr_size=1,
            out_act="softmax",
            config=config,
            **kwargs
        )

    def loss(self, y_true, y_pred, weights):
        return weighted_cross_entropy(y_pred, y_true, weights=weights)


class DeterministicEpidemicsWGNN(DeterministicEpidemicsGNN, WeightedGraphNeuralNetwork):
    def __init__(self, config=None, **kwargs):
        DeterministicEpidemicsGNN.__init__(self, config=config, *kwargs)
        WeightedGraphNeuralNetwork.__init__(
            self,
            self.config.num_states,
            self.config.num_states,
            window_size=self.config.window_size,
            nodeattr_size=1,
            edgeattr_size=1,
            out_act="softmax",
            config=config,
            **kwargs
        )


class DeterministicEpidemicsMGNN(
    DeterministicEpidemicsGNN, MultiplexGraphNeuralNetwork
):
    def __init__(self, config=None, **kwargs):
        DeterministicEpidemicsGNN.__init__(self, config=config, *kwargs)
        MultiplexGraphNeuralNetwork.__init__(
            self,
            self.config.num_states,
            self.config.num_states,
            window_size=self.config.window_size,
            nodeattr_size=1,
            out_act="softmax",
            config=config,
            **kwargs
        )


class DeterministicEpidemicsWMGNN(
    DeterministicEpidemicsGNN, WeightedMultiplexGraphNeuralNetwork
):
    def __init__(self, config=None, **kwargs):
        DeterministicEpidemicsGNN.__init__(self, config=config, *kwargs)
        WeightedMultiplexGraphNeuralNetwork.__init__(
            self,
            self.config.num_states,
            self.config.num_states,
            window_size=self.config.window_size,
            nodeattr_size=1,
            edgeattr_size=1,
            out_act="softmax",
            config=config,
            **kwargs
        )
