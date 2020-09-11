import torch
import torch.nn as nn

from .gnn import (
    GraphNeuralNetwork,
    WeightedGraphNeuralNetwork,
    MultiplexGraphNeuralNetwork,
    WeightedMultiplexGraphNeuralNetwork,
)
from dynalearn.config import Config
from dynalearn.nn.loss import weighted_mse


class ReactionDiffusionGNN(GraphNeuralNetwork):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        self.num_states = config.num_states
        self.alpha = config.alpha
        GraphNeuralNetwork.__init__(
            self,
            config.num_states,
            config.num_states,
            window_size=config.window_size,
            normalize=True,
            config=config,
            **kwargs
        )

    def loss(self, y_true, y_pred, weights):
        l1 = weighted_mse(y_true, y_pred, weights=weights)
        sizes_true = torch.sum(y_true, axis=-1)
        sizes_pred = torch.sum(y_pred, axis=-1)
        l2 = torch.sum(weights * torch.abs(sizes_true - sizes_pred))
        return self.alpha[0] * l1 + self.alpha[0] * l2


class ReactionDiffusionWGNN(ReactionDiffusionGNN, WeightedGraphNeuralNetwork):
    def __init__(self, config=None, **kwargs):
        ReactionDiffusionGNN.__init__(self, config=config, **kwargs)
        WeightedGraphNeuralNetwork.__init__(
            self,
            self.config.num_states,
            self.config.num_states,
            window_size=self.config.window_size,
            edgeatttr_size=1,
            normalize=True,
            config=config,
            **kwargs
        )


class ReactionDiffusionMGNN(ReactionDiffusionGNN, MultiplexGraphNeuralNetwork):
    def __init__(self, config=None, **kwargs):
        ReactionDiffusionGNN.__init__(self, config=config, **kwargs)
        MultiplexGraphNeuralNetwork.__init__(
            self,
            self.config.num_states,
            self.config.num_states,
            window_size=self.config.window_size,
            normalize=True,
            config=config,
            **kwargs
        )


class ReactionDiffusionWMGNN(ReactionDiffusionGNN, WeightedMultiplexGraphNeuralNetwork):
    def __init__(self, config=None, **kwargs):
        ReactionDiffusionGNN.__init__(self, config=config, **kwargs)
        WeightedMultiplexGraphNeuralNetwork.__init__(
            self,
            self.config.num_states,
            self.config.num_states,
            window_size=self.config.window_size,
            edgeattr_size=1,
            normalize=True,
            config=config,
            **kwargs
        )
