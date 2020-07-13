from dynalearn.dynamics.metapopulation import MetaPopulationDynamics
from dynalearn.nn.callbacks import CallbackList
from dynalearn.nn.history import History
from dynalearn.nn.loss import get as get_loss
from dynalearn.nn.models import MetaPopGNN
from dynalearn.nn.optimizer import get as get_optimizer
from dynalearn.utilities import to_edge_index
from dynalearn.config import Config


import networkx as nx
import numpy as np
import time
import torch


class TrainableMetapopulation(MetaPopulationDynamics):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwagrs
        self.nn = MetaPopulationGNN(config)
        if torch.cuda.is_available():
            self.nn = self.nn.cuda()
        DynamicsModel.__init__(self, config, config.num_states)

    def initial_state(self):
        return np.random.randint(self.num_states, size=(self.num_nodes,))

    def is_dead(self):
        return False

    def sample(self, x):
        return self.predict(x)

    def predict(self, x):
        if type(x) == np.ndarray:
            x = torch.Tensor(x)
        edge_index = self.edge_index
        if torch.cuda.is_available():
            x = x.cuda()
            edge_index = self.edge_index.cuda()
        return self.nn.forward(x, edge_index).cpu().detach().numpy()
