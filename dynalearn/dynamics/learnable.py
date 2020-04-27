from dynalearn.dynamics.base import DynamicsModel
from dynalearn.nn.callbacks import CallbackList
from dynalearn.nn.history import History
from dynalearn.nn.loss import get as get_loss
from dynalearn.nn.models import GeneralEpidemicsGNN
from dynalearn.nn.optimizer import get as get_optimizer
from dynalearn.utilities import to_edge_index
from dynalearn.config import Config


import networkx as nx
import numpy as np
import time
import torch


class LearnableEpidemics(DynamicsModel):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwagrs
        self.nn = GeneralEpidemicsGNN(config)
        if torch.cuda.is_available():
            self.nn = self.nn.cuda()
        self._edge_index = None
        DynamicsModel.__init__(self, config, config.num_states)

    def initial_state(self):
        return np.random.randint(self.num_states, size=(self.num_nodes,))

    def is_dead(self):
        return False

    def sample(self, x):
        p = self.predict(x)
        dist = torch.distributions.Categorical(torch.tensor(p))
        return np.array(dist.sample())

    def predict(self, x):
        if type(x) == np.ndarray:
            x = torch.Tensor(x)
        edge_index = self.edge_index
        if torch.cuda.is_available():
            x = x.cuda()
            edge_index = self.edge_index.cuda()
        return self.nn.forward(x, edge_index).cpu().detach().numpy()

    @property
    def network(self):
        if self._network is None:
            raise ValueError("No network has been parsed to the dynamics.")
        else:
            return self._network

    @network.setter
    def network(self, network):
        self._network = network
        if not network.is_directed():
            network = nx.to_directed(network)
        self._edge_index = to_edge_index(network)
        self._num_nodes = self._network.number_of_nodes()

    @property
    def edge_index(self):
        if self._edge_index is None:
            raise ValueError("No network has been parsed to the dynamics.")
        else:
            return self._edge_index
