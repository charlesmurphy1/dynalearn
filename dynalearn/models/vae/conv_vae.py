import numpy as np
import torch.nn as nn
from .vae import VAE


class Conv_Encoder(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_embedding, n_channel=1):
        super(Conv_Encoder, self).__init__()
        
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_embedding = n_embedding
        self.n_channel = n_channel

        # Functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.maxpool2 = nn.MaxPool2d(2)
        self.maxpool7 = nn.MaxPool2d(7)

        # Encoder: Inference network weights
        self.conv1 = nn.Conv2d(self.n_channel, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        self.fc1 = nn.Linear(128, self.n_hidden)
        
        # Embedding networks weights
        self.mu = nn.Linear(self.n_hidden,
                            self.n_embedding)
        self.logvar = nn.Linear(self.n_hidden,
                                self.n_embedding)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.maxpool2(self.relu(self.conv1(x)))  # from 28 * 28 to 14 * 14
        x = self.maxpool2(self.relu(self.conv2(x)))  # from 14 * 14 to 7 * 7
        x = self.maxpool7(self.relu(self.conv3(x)))  # from 7 * 7 to 1 * 1
        x = x.view(batch_size, 128)
        x = self.relu(self.fc1(x))

        return self.mu(x), self.logvar(x)


class Conv_Decoder(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_embedding, n_channel=1):
        super(Conv_Decoder, self).__init__()

        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_embedding = n_embedding
        self.n_channel = n_channel

        # Functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.maxunpool2 = nn.MaxUnpool2d(2)
        self.maxunpool7 = nn.MaxUnpool2d(7)
    
        # Decoder: Generative network weights
        self.fc1 = nn.Linear(n_embedding, 128)
        self.convTrans1 = nn.ConvTranspose2d(32, 1, 5, padding=2)
        self.convTrans2 = nn.ConvTranspose2d(64, 32, 3, padding=1)
        self.convTrans3 = nn.ConvTranspose2d(128, 64, 3, padding=1)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.relu(self.fc1(x))
        x = x.view(self.batch_size, 128)
        x = self.maxunpool7(self.relu(self.convTrans3(x)))
        x = self.maxunpool2(self.relu(self.convTrans2(x)))
        x = self.maxunpool2(self.relu(self.convTrans1(x)))
        return x


def Conv_VAE(n_inputs, n_hidden, n_embedding, n_channel=1, optimizer=None,
             loss=None, use_cuda=False):
    encoder = Conv_Encoder(n_inputs, n_hidden, n_embedding, n_channel)
    decoder = Conv_Decoder(n_inputs, n_hidden, n_embedding, n_channel)
    return VAE(encoder, decoder, n_embedding,
               optimizer=optimizer,
               loss=loss,
               use_cuda=use_cuda)