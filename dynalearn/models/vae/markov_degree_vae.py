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
    def __init__(self, graph, encoder, decoder, optimizer=None, loss=None,
                 scheduler=None, use_cuda=False):
        super(MarkovDegreeVAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.loss = loss
        self.scheduler = scheduler
        self.use_cuda = use_cuda

        self.set_graph(graph)
        self.kmax = self.encoder.kmax
        self.n_embedding = self.encoder.n_embedding

        if self.use_cuda:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()

        self.setup_trainer(optimizer, loss, scheduler)

        
    def _get_embedding_size(self):
        return torch.Size([self.n_embedding, self.num_nodes])


    def _get_past_size(self):
        return torch.Size([1, self.num_nodes])


    def _format_present(self, present):
        batch_size = present.size(0)
        return present.contiguous().view(batch_size, 1, self.num_nodes)


    def _format_past(self, past):
        batch_size = past.size(0)
        past = past.contiguous().view(batch_size, 1, self.num_nodes)
        _past = past.repeat(1, self.num_nodes, 1)
        _past.masked_fill_(self.edgeMask, 0)
        inf_degree = torch.sum(_past, 1).contiguous().view([batch_size, 1,
                                                            self.num_nodes])
        degree_onehot = torch.zeros(batch_size, self.kmax + 1, self.num_nodes)
        degree_onehot.scatter_(1, inf_degree.long(), 1).float()

        return torch.cat([past, degree_onehot], 1)


    def set_graph(self, graph):
        self.num_nodes = graph.number_of_nodes()
        adjacency_matrix = torch.from_numpy(nx.to_numpy_array(graph,
                                            nodelist=range(self.num_nodes))).int()

        self.edgeMask = 1 - adjacency_matrix.byte()


def basicMarkovDegreeVAE(graph, n_hidden, n_embedding, kmax=None,
                        keepprob=1, optimizer=None,
                        loss=None, scheduler=None,
                        use_cuda=False):

    encoder = NodeDegreeEncoder(graph, n_hidden, n_embedding, kmax, keepprob)
    decoder = NodeDegreeDecoder(graph, n_hidden, n_embedding, kmax, keepprob)

    return MarkovDegreeVAE(graph, encoder, decoder, optimizer, loss, scheduler, use_cuda)