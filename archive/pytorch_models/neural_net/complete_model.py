import numpy as np
from random import randint
import sys
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .general_model import MarkovNN
from .layers import CompleteModel, CompleteMultiStateModel


class CompleteMarkovNN(MarkovNN):
    def __init__(self, model, use_cuda=False):
        super(CompleteMarkovNN, self).__init__()

        self.model = model
        self.use_cuda = use_cuda

        if self.use_cuda:
            self.model = self.model.cuda()




def basicCompleteMarkovNN(graph, n_hidden, use_cuda=False):
    model = CompleteModel(graph, n_hidden)

    return CompleteMarkovNN(model, use_cuda=use_cuda)


def basicCompleteMarkovNNMultistate(graph, n_hidden, n_states, use_cuda=False):
    model = CompleteMultiStateModel(graph, n_hidden, n_states)
    return CompleteMarkovNN(model, use_cuda=use_cuda)