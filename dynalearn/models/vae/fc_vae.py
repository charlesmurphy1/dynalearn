import numpy as np
import torch.nn as nn
from .vae import VAE


class Fc_Encoder(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_embedding, keepprob=1):
        super(Fc_Encoder, self).__init__()
        if type(n_hidden) == int: n_hidden = [n_hidden]
    
        # Functions
        relu = nn.ReLU()
        tanh = nn.Tanh()
        sigmoid = nn.Sigmoid()
        dropout = nn.Dropout(1 - keepprob)
        
        # Encoder: Inference network weights
        layers = [nn.Linear(n_inputs, n_hidden[0])]
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

    def forward(self, x):
        x = self.encoder(x)
        return self.mu(x), self.var(x)


class Fc_Decoder(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_embedding, keepprob=1):
        super(Fc_Decoder, self).__init__()
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
        self.decoder = nn.Sequential(nn.Linear(n_embedding,
                                                n_hidden[-1]),
                                      tanh,
                                      dropout,
                                      *layers,
                                      nn.Linear(n_hidden[0],
                                                n_inputs),
                                      sigmoid)

    def forward(self, z):
        return self.decoder(z)


def FC_VAE(n_inputs, n_hidden, n_embedding, keepprob,
           loss=None, optimizer=None, use_cuda=False):
    encoder = Fc_Encoder(n_inputs, n_hidden, n_embedding, keepprob)
    decoder = Fc_Decoder(n_inputs, n_hidden, n_embedding, keepprob)
    return VAE(encoder, decoder, n_embedding,
               optimizer=optimizer, loss=loss, use_cuda=use_cuda)