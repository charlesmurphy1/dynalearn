import torch
import torch.nn as nn

from .gnn import (
    GraphNeuralNetwork,
    WeightedGraphNeuralNetwork,
    MultiplexGraphNeuralNetwork,
    WeightedMultiplexGraphNeuralNetwork,
)
from .multivariate import (
    MultivariateMPL,
    MultivariateRNN,
)
from .univariate import (
    UnivariateMPL,
    UnivariateRNN,
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
        return weighted_cross_entropy(y_true, y_pred, weights=weights)


class StochasticEpidemicsWGNN(StochasticEpidemicsGNN, WeightedGraphNeuralNetwork):
    def __init__(self, config=None, **kwargs):
        StochasticEpidemicsGNN.__init__(self, config=config, **kwargs)
        WeightedGraphNeuralNetwork.__init__(
            self,
            1,
            self.config.num_states,
            window_size=self.config.window_size,
            edgeattr_size=1,
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


class StochasticEpidemicsUMPL(UnivariateMPL):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        self.num_states = config.num_states
        UnivariateMPL.__init__(
            self,
            1,
            config.num_states,
            window_size=config.window_size,
            out_act="softmax",
            config=config,
            **kwargs
        )

    def loss(self, y_true, y_pred, weights):
        return weighted_cross_entropy(y_true, y_pred, weights=weights)


class StochasticEpidemicsURNN(UnivariateRNN):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        self.num_states = config.num_states
        UnivariateRNN.__init__(
            self,
            1,
            config.num_states,
            window_size=config.window_size,
            out_act="softmax",
            rnn=config.rnn,
            config=config,
            **kwargs
        )

    def loss(self, y_true, y_pred, weights):
        return weighted_cross_entropy(y_true, y_pred, weights=weights)


class StochasticEpidemicsMMPL(MultivariateMPL):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        self.num_states = config.num_states
        MultivariateMPL.__init__(
            self,
            1,
            config.num_states,
            config.num_nodes,
            window_size=config.window_size,
            out_act="softmax",
            config=config,
            **kwargs
        )

    def loss(self, y_true, y_pred, weights):
        return weighted_cross_entropy(y_true, y_pred, weights=weights)


class StochasticEpidemicsMRNN(MultivariateRNN):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        self.num_states = config.num_states
        MultivariateRNN.__init__(
            self,
            1,
            config.num_states,
            config.num_nodes,
            window_size=config.window_size,
            out_act="softmax",
            rnn=config.rnn,
            config=config,
            **kwargs
        )

    def loss(self, y_true, y_pred, weights):
        return weighted_cross_entropy(y_true, y_pred, weights=weights)
