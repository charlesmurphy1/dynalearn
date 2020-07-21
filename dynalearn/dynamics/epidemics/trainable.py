import networkx as nx
import numpy as np
import time
import torch

from dynalearn.dynamics.epidemics import Epidemics
from dynalearn.nn.callbacks import CallbackList
from dynalearn.nn.history import History
from dynalearn.nn.loss import get as get_loss
from dynalearn.nn.models import EpidemicsGNN  # , WithNonEdgeEpidemicsGNN
from dynalearn.nn.optimizer import get as get_optimizer
from dynalearn.utilities import to_edge_index
from dynalearn.config import Config


class TrainableEpidemics(Epidemics):
    def __init__(self, config=None, **kwargs):
        Epidemics.__init__(self, config, config.num_states)
        if config is None:
            config = Config()
            config.__dict__ = kwagrs
        self.window_size = config.window_size
        self.window_step = config.window_step
        self.nn = EpidemicsGNN(config)
        if torch.cuda.is_available():
            self.nn = self.nn.cuda()

    def initial_state(self):
        return np.random.randint(
            self.num_states, size=(self.window_size, self.num_nodes)
        ).squeeze()

    def is_dead(self):
        return False

    def predict(self, x):
        if type(x) == np.ndarray:
            x = torch.Tensor(x)
        edge_index = self.edge_index
        if torch.cuda.is_available():
            x = x.cuda()
            edge_index = self.edge_index.cuda()
        return self.nn.forward(x, edge_index).cpu().detach().numpy()
