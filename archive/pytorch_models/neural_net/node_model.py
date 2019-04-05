import numpy as np
from random import randint
import sys
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .general_model import MarkovNN
from .layers import GCNModel, GATModel


class NodeMarkovNN(MarkovNN):
    def __init__(self, model, use_cuda=False):
        super(NodeMarkovNN, self).__init__()

        self.model = model
        self.use_cuda = use_cuda

        if self.use_cuda:
            self.model = self.model.cuda()


def basicNodeMarkovGCN(n_hidden, use_cuda=False):
    model = GCNModel(graph, n_hidden)

    return NodeMarkovNN(model, use_cuda=use_cuda)


def basicNodeMarkovGAT(n_hidden, n_heads, dropout, alpha,
                       min_value=1e-15, with_self_attn=False,
                       use_cuda=False):
    model = GATModel(n_hidden, n_heads, dropout, alpha, min_value,
                     with_self_attn)
    return NodeMarkovNN(model, use_cuda=use_cuda)