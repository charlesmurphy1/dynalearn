import numpy as np
import torch
import torch.nn as nn
from .cvae import CVAE


class Fc_CEncoder(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_embedding, n_conditional, 
                 keepprob=1):
        super(Fc_CEncoder, self).__init__()
        if type(n_hidden) == int: n_hidden = [n_hidden]
    
        # Functions
        relu = nn.ReLU()
        tanh = nn.Tanh()
        sigmoid = nn.Sigmoid()
        dropout = nn.Dropout(1 - keepprob)
        
        # Encoder: Inference network weights
        layers = [nn.Linear(n_inputs + n_conditional, n_hidden[0])]
        for i in range(1, len(n_hidden)):
            layers.append(relu)
            layers.append(dropout)
            layers.append(nn.Linear(n_hidden[i - 1], n_hidden[i]))
        self.encoder = nn.Sequential(*layers, tanh, dropout)
        
        # Embedding networks weights
        self.mu = nn.Linear(n_hidden[-1],
                             n_embedding)
        self.var = nn.Linear(n_hidden[-1],
                                 n_embedding)

    def forward(self, x, c):
        x = torch.cat([x, c], 1)
        h = self.encoder(x)
        return self.mu(h), self.var(h)


class Fc_CDecoder(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_embedding, n_conditional,
                 keepprob=1):
        super(Fc_CDecoder, self).__init__()
        if type(n_hidden) == int: n_hidden = [n_hidden]

        # Functions
        relu = nn.ReLU()
        tanh = nn.Tanh()
        sigmoid = nn.Sigmoid()
        dropout = nn.Dropout(1 - keepprob)
    
        # Decoder: Generative network weights
        layers = []
        for i in range(len(n_hidden) - 1, 0, -1):
            layers.append(nn.Linear(n_hidden[i],
                                    n_hidden[i - 1]))
            layers.append(relu)
            layers.append(dropout)
        self.decoder = nn.Sequential(nn.Linear(n_embedding + n_conditional,
                                                n_hidden[-1]),
                                      tanh,
                                      dropout,
                                      *layers,
                                      nn.Linear(n_hidden[0],
                                                n_inputs),
                                      sigmoid)

    def forward(self, z, c):
        z = torch.cat([z, c], 1)

        return self.decoder(z)


def FC_CVAE(n_inputs, n_hidden, n_embedding, n_conditional, keepprob,
            optimizer=None, use_cuda=False):
    encoder = Fc_CEncoder(n_inputs, n_hidden, n_embedding, n_conditional, keepprob)
    decoder = Fc_CDecoder(n_inputs, n_hidden, n_embedding, n_conditional, keepprob)
    return CVAE(encoder, decoder, n_embedding, n_conditional,
                optimizer=optimizer, use_cuda=use_cuda)