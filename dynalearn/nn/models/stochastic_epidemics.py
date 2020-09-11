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


class StochasticEpidemicsGNN(GraphNeuralNetwork):
    def __init__(self, config=None, **kwargs):
        config = config or Config(**kwargs)
        self.num_states = config.num_states
        GraphNeuralNetwork.__init__(
            self,
            1,
            config.num_states,
            window_size=config.window_size,
            out_act="softmax",
            config=config,
            **kwargs
        )

    def loss(self, y_true, y_pred, weights):
        return weighted_cross_entropy(y_pred, y_true, weights=weights)


class StochasticEpidemicsWGNN(StochasticEpidemicsGNN, WeightedGraphNeuralNetwork):
    def __init__(self, config=None, **kwargs):
        StochasticEpidemicsGNN.__init__(self, config=config, **kwargs)
        WeightedGraphNeuralNetwork.__init__(
            self,
            1,
            self.config.num_states,
            window_size=self.config.window_size,
            edgeatttr_size=1,
            out_act="softmax",
            config=config,
            **kwargs
        )


class StochasticEpidemicsMGNN(StochasticEpidemicsGNN, MultiplexGraphNeuralNetwork):
    def __init__(self, config=None, **kwargs):
        StochasticEpidemicsGNN.__init__(self, config=config, **kwargs)
        MultiplexGraphNeuralNetwork.__init__(
            self,
            1,
            self.config.num_states,
            window_size=self.config.window_size,
            out_act="softmax",
            config=config,
            **kwargs
        )


class StochasticEpidemicsWMGNN(
    StochasticEpidemicsGNN, WeightedMultiplexGraphNeuralNetwork
):
    def __init__(self, config=None, **kwargs):
        StochasticEpidemicsGNN.__init__(self, config=config, **kwargs)
        WeightedMultiplexGraphNeuralNetwork.__init__(
            self,
            1,
            self.config.num_states,
            window_size=self.config.window_size,
            edgeattr_size=1,
            out_act="softmax",
            config=config,
            **kwargs
        )
