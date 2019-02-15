import numpy as np
from random import randint
import sys
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .markov_vae import MarkovVAE
from .markov_layers import CompleteEncoder, CompleteDecoder


class MarkovCompleteVAE(MarkovVAE):
    def __init__(self, encoder, decoder, optimizer=None, loss=None,
                 scheduler=None, use_cuda=False):
        super(MarkovCompleteVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.use_cuda = use_cuda

        self.num_nodes = self.encoder.num_nodes
        self.n_embedding = self.encoder.n_embedding

        if self.use_cuda:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()

        self.setup_trainer(optimizer, loss, scheduler)
        
    def _get_embedding_size(self):
        return torch.Size([self.n_embedding])


    def _get_past_size(self):
        return torch.Size([self.num_nodes])



def basicMarkovCompleteVAE(graph, n_hidden, n_embedding, keepprob=1,
                           optimizer=None, loss=None, scheduler=None,
                           use_cuda=False):
    encoder = CompleteEncoder(graph, n_hidden, n_embedding, keepprob)
    decoder = CompleteDecoder(graph, n_hidden, n_embedding, keepprob)

    return MarkovCompleteVAE(encoder, decoder, 
                             optimizer=optimizer,
                             loss=loss,
                             scheduler=scheduler,
                             use_cuda=use_cuda)