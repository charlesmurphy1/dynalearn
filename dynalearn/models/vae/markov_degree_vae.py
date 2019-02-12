import networkx as nx
import numpy as np
from random import randint
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F


from .markov_vae import MarkovVAE
from .markov_layers import NodeDegreeEncoder, NodeDegreeDecoder

class MarkovDegreeVAE(MarkovVAE):
    def __init__(self, encoder, decoder, optimizer=None, loss=None,
                 use_cuda=False):
        super(MarkovDegreeVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.loss = loss
        self.use_cuda = use_cuda

        self.num_nodes = self.encoder.num_nodes
        self.kmax = self.encoder.kmax
        self.n_embedding = self.encoder.n_embedding

        if self.use_cuda:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()

        if optimizer is None:
            self.optimizer = torch.optim.SGD(self.parameters(), lr = 1e-3)
        else:
            self.optimizer = optimizer(self.parameters())


        if loss is None:
            self.loss = nn.BCELoss(reduction="sum")
        else:
            self.loss = loss
        
    def _get_embedding_size(self):
        return torch.Size([self.n_embedding, self.num_nodes])


    def _get_past_size(self):
        return torch.Size([1, self.num_nodes])


    def _get_conditional(self, past):
        batch_size = past.size(0)
        cond_onehot = torch.zeros(batch_size, self.kmax, self.num_nodes)
        cond_onehot.scatter_(1, past.long(), 1).float()

        return torch.cat([past, cond_onehot], 1)


def basicMarkovDegreeVAE(graph, n_hidden, n_embedding,
                        keepprob=1, optimizer=None,
                        loss=None, use_cuda=False):

    encoder = NodeDegreeEncoder(graph, n_hidden, n_embedding, keepprob)
    decoder = NodeDegreeDecoder(graph, n_hidden, n_embedding, keepprob)

    return MarkovDegreeVAE(encoder, decoder, optimizer, loss, use_cuda)