import numpy as np
from random import randint
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F

from .markov_layers import NodeEncoder, NodeDecoder, SparseNodeEncoder
from .markov_vae import MarkovVAE

class MarkovNodeVAE(MarkovVAE):
    def __init__(self, encoder, decoder, optimizer=None, loss=None,
                 use_cuda=False):
        super(MarkovNodeVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.loss = loss
        self.use_cuda = use_cuda

        self.num_nodes = self.encoder.num_nodes
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


    def _format_present(self, present):
        batch_size = present.size(0)
        return present.view(batch_size, 1, self.num_nodes)

    def _format_past(self, past):
        batch_size = past.size(0)
        past = past.view(batch_size, 1, self.num_nodes)
        return past.repeat(1, self.num_nodes, 1)


def basicMarkovNodeVAE(graph, n_hidden, n_embedding,
                        keepprob=1, optimizer=None,
                        loss=None, use_cuda=False):

    encoder = NodeEncoder(graph, n_hidden, n_embedding, keepprob)
    decoder = NodeDecoder(graph, n_hidden, n_embedding, keepprob)

    return MarkovNodeVAE(encoder, decoder, optimizer, loss, use_cuda)


def sparseMarkovNodeVAE(graph, n_hidden, n_embedding,
                         keepprob=1, optimizer=None,
                         loss=None, use_cuda=False):

    encoder = SparseNodeEncoder(graph, n_hidden, n_embedding, keepprob)
    decoder = NodeDecoder(graph, n_hidden, n_embedding, keepprob)

    return MarkovNodeVAE(encoder, decoder, optimizer, loss, use_cuda)
