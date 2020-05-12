import numpy as np
import torch

from torch_geometric.nn.conv import MessagePassing
from dynalearn.utilities import onehot


class Propagator(MessagePassing):
    def __init__(self, num_states):
        MessagePassing.__init__(self, aggr="add")
        self.num_states = num_states

    def forward(self, x, edge_index):
        if type(x) == np.ndarray:
            x = torch.Tensor(x)
        if torch.cuda.is_available():
            x = x.cuda()
            edge_index = edge_index.cuda()

        x = onehot(x, num_class=self.num_states)
        return self.propagate(edge_index, x=x).T

    def message(self, x_j):
        return x_j

    def update(self, x):
        return x
