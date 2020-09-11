import numpy as np
import time
import torch

from dynalearn.dynamics.stochastic_epidemics import StochasticEpidemics
from dynalearn.nn.models import (
    StochasticEpidemicsGNN,
    StochasticEpidemicsWGNN,
    StochasticEpidemicsMGNN,
    StochasticEpidemicsWMGNN,
)
from dynalearn.config import Config


class SimpleTrainableStochasticEpidemics(StochasticEpidemics):
    def __init__(self, config=None, **kwargs):
        self.config = config or Config(**kwargs)
        StochasticEpidemics.__init__(self, config, config.num_states)
        self.window_size = config.window_size
        self.window_step = config.window_step
        self.nn = StochasticEpidemicsGNN(config)
        if torch.cuda.is_available():
            self.nn = self.nn.cuda()

    def initial_state(self):
        return np.random.randint(
            self.num_states, size=(self.num_nodes, self.window_size)
        ).squeeze()

    def is_dead(self):
        return False

    def predict(self, x):
        if isinstance(x, np.ndarray):
            x = torch.Tensor(x)
        assert x.ndim == 2
        assert x.shape[-1] == self.window_size
        x = self.nn.transformers["t_inputs"].forward(x)
        g = self.nn.transformers["t_networks"].forward(self.network)
        y = self.nn.transformers["t_targets"].backward(self.nn.forward(x, g))
        return y.cpu().detach().numpy()


def TrainableStochasticEpidemics(config=None, **kwargs):
    return SimpleTrainableStochasticEpidemics(config=config, **kwargs)
