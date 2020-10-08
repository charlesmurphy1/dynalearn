import networkx as nx
import numpy as np
import time
import torch

from dynalearn.dynamics.deterministic_epidemics import (
    DeterministicEpidemics,
    WeightedDeterministicEpidemics,
    MultiplexDeterministicEpidemics,
    WeightedMultiplexDeterministicEpidemics,
)
from dynalearn.nn.models import (
    DeterministicEpidemicsGNN,
    DeterministicEpidemicsWGNN,
    DeterministicEpidemicsMGNN,
    DeterministicEpidemicsWMGNN,
)
from dynalearn.nn.optimizers import get as get_optimizer
from dynalearn.utilities import to_edge_index
from dynalearn.config import Config


class SimpleTrainableDeterministicEpidemics(DeterministicEpidemics):
    def __init__(self, config=None, **kwargs):
        self.config = config or Config(**kwargs)
        DeterministicEpidemics.__init__(self, config, config.num_states)
        # self.window_size = config.window_size
        # self.window_step = config.window_step
        self.nn = DeterministicEpidemicsGNN(config)
        if torch.cuda.is_available():
            self.nn = self.nn.cuda()

    def initial_state(self):
        x = np.random.rand(self.num_nodes, self.num_states, self.window_size)
        x /= x.sum(1).reshape(self.num_nodes, 1, self.window_size)
        return x

    def is_dead(self):
        return False

    def update(self, x):
        return x

    def predict(self, x):
        if isinstance(x, np.ndarray):
            x = torch.Tensor(x)
        assert x.ndim == 3
        assert x.shape[1] == self.num_states
        assert x.shape[2] == self.window_size
        x = self.nn.transformers["t_inputs"].forward(x)
        g = self.nn.transformers["t_networks"].forward(self.network)
        y = self.nn.transformers["t_targets"].backward(self.nn.forward(x, g))
        return y.cpu().detach().numpy()


class WeightedTrainableDeterministicEpidemics(
    SimpleTrainableDeterministicEpidemics, WeightedDeterministicEpidemics
):
    def __init__(self, config=None, **kwargs):
        config = config or Config(**kwargs)
        WeightedDeterministicEpidemics.__init__(self, config, config.num_states)
        SimpleTrainableDeterministicEpidemics.__init__(self, config=config, **kwargs)
        self.nn = DeterministicEpidemicsWGNN(config)


class MultiplexTrainableDeterministicEpidemics(
    SimpleTrainableDeterministicEpidemics, MultiplexDeterministicEpidemics
):
    def __init__(self, config=None, **kwargs):
        config = config or Config(**kwargs)
        MultiplexDeterministicEpidemics.__init__(self, config, config.num_states)
        SimpleTrainableDeterministicEpidemics.__init__(self, config=config, **kwargs)
        self.nn = DeterministicEpidemicsMGNN(config)


class WeightedMultiplexTrainableDeterministicEpidemics(
    SimpleTrainableDeterministicEpidemics, WeightedMultiplexDeterministicEpidemics
):
    def __init__(self, config=None, **kwargs):
        config = config or Config(**kwargs)
        WeightedMultiplexDeterministicEpidemics.__init__(
            self, config, config.num_states
        )
        SimpleTrainableDeterministicEpidemics.__init__(self, config=config, **kwargs)
        self.nn = DeterministicEpidemicsWMGNN(config)


def TrainableDeterministicEpidemics(config=None, **kwargs):
    config = config or Config(**kwargs)
    if "is_weighted" in config.__dict__:
        is_weighted = config.is_weighted
    else:
        is_weighted = False

    if "is_multiplex" in config.__dict__:
        is_multiplex = config.is_multiplex
    else:
        is_multiplex = False

    if is_weighted and is_multiplex:
        return WeightedMultiplexTrainableDeterministicEpidemics(config=config, **kwargs)
    elif is_weighted and not is_multiplex:
        return WeightedTrainableDeterministicEpidemics(config=config, **kwargs)
    elif not is_weighted and is_multiplex:
        return MultiplexTrainableDeterministicEpidemics(config=config, **kwargs)
    else:
        return SimpleTrainableDeterministicEpidemics(config=config, **kwargs)
