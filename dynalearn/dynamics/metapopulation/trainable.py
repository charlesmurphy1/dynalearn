import networkx as nx
import numpy as np
import time
import torch

from dynalearn.dynamics.metapopulation import (
    MetaPop,
    WeightedMetaPop,
    MultiplexMetaPop,
    WeightedMultiplexMetaPop,
)
from dynalearn.nn.callbacks import CallbackList
from dynalearn.nn.history import History
from dynalearn.nn.loss import get as get_loss
from dynalearn.nn.models import MetaPopGNN, MetaPopWGNN, MetaPopMGNN, MetaPopWMGNN
from dynalearn.nn.optimizer import get as get_optimizer
from dynalearn.utilities import to_edge_index
from dynalearn.config import Config


class SimpleTrainableMetaPop(MetaPop):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwagrs
        MetaPop.__init__(self, config, config.num_states)
        self.nn = MetaPopGNN(config)
        self.window_size = config.window_size
        self.window_step = config.window_step
        if torch.cuda.is_available():
            self.nn = self.nn.cuda()

    def reaction(self, x):
        return x

    def diffusion(self, x):
        return x

    def initial_state(self):
        return np.random.randint(self.num_states, size=(self.num_nodes,))

    def is_dead(self):
        return False

    def sample(self, x):
        return self.predict(x)

    def predict(self, x):
        if isinstance(x, np.ndarray):
            x = torch.Tensor(x)
        edge_index = self.edge_index
        if torch.cuda.is_available():
            x = x.cuda()
            edge_index = edge_index.cuda()
        x = self.nn.normalize(x, "inputs")
        y = self.nn.forward(x, edge_index)
        y = self.nn.unnormalize(y, "targets")
        return y.cpu().detach().numpy()


class WeightedTrainableMetaPop(SimpleTrainableMetaPop, WeightedMetaPop):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwagrs
        WeightedMetaPop.__init__(self, config, config.num_states)
        SimpleTrainableMetaPop.__init__(self, config=config, **kwargs)
        self.nn = MetaPopWGNN(config)

    def predict(self, x):
        if isinstance(x, np.ndarray):
            x = torch.Tensor(x)
        edge_index = self.edge_index
        edge_weight = torch.Tensor(self.edge_weight)
        if torch.cuda.is_available():
            x = x.cuda()
            edge_index = edge_index.cuda()
            edge_weight = edge_weight.cuda()
        x = self.nn.normalize(x, "inputs")
        edge_weight = self.nn.normalize(edge_weight, "edge_attr")
        y = self.nn.forward(x, edge_index, edge_attr=edge_weight)
        y = self.nn.unnormalize(y, "targets")
        return y.cpu().detach().numpy()


class MultiplexTrainableMetaPop(SimpleTrainableMetaPop, MultiplexMetaPop):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwagrs

        MultiplexMetaPop.__init__(self, config, config.num_states)
        SimpleTrainableMetaPop.__init__(self, config=config, **kwargs)
        self.nn = MetaPopMGNN(config)


class WeightedMultiplexTrainableMetaPop(
    SimpleTrainableMetaPop, WeightedMultiplexMetaPop
):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwagrs

        WeightedMultiplexMetaPop.__init__(self, config, config.num_states)
        SimpleTrainableMetaPop.__init__(self, config=config, **kwargs)
        self.nn = MetaPopWMGNN(config)

    def predict(self, x):
        if isinstance(x, np.ndarray):
            x = torch.Tensor(x)
        edge_index = self.edge_index
        edge_weight = {
            k: torch.Tensor(w).reshape(-1, 1) for k, w in self.edge_weight.items()
        }
        if torch.cuda.is_available():
            x = x.cuda()
            edge_index = edge_index.cuda()
            edge_weight = {k: w.cuda() for k, w in self.edge_weight.items()}
        if x.ndim == 2:
            x = x.view(-1, self.num_states, 1)
        x = self.nn.normalize(x, "inputs")
        edge_weight = self.nn.normalize(edge_weight, "edge_attr")
        y = self.nn.forward(x, edge_index, edge_attr=edge_weight)
        y = self.nn.unnormalize(y, "targets")
        return y.cpu().detach().numpy()


def TrainableMetaPop(config=None, **kwargs):
    if config is None:
        config = Config()
        config.__dict__ = kwagrs

    if "is_weighted" in config.__dict__:
        is_weighted = config.is_weighted
    else:
        is_weighted = False

    if "is_multiplex" in config.__dict__:
        is_multiplex = config.is_multiplex
    else:
        is_multiplex = False

    if is_weighted and is_multiplex:
        return WeightedMultiplexTrainableMetaPop(config=config, **kwargs)
    elif is_weighted and not is_multiplex:
        return WeightedTrainableMetaPop(config=config, **kwargs)
    elif not is_weighted and is_multiplex:
        return MultiplexTrainableMetaPop(config=config, **kwargs)
    else:
        return SimpleTrainableMetaPop(config=config, **kwargs)
