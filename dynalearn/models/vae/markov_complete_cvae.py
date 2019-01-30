import numpy as np
from random import randint
import progressbar
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .fc_cvae import Fc_CEncoder, Fc_CDecoder
from .cvae import CVAE

import time


class Markov_Complete_CVAE(nn.Module):
    def __init__(self, n_nodes, n_hidden, n_embedding,
                 optimizer=None, loss=None, use_cuda=False):
        super(Markov_Complete_CVAE, self).__init__()
        
        self.n_nodes = n_nodes
        self.n_hidden = n_hidden
        self.n_embedding = n_embedding
        self.use_cuda = use_cuda

        self.encoder = Fc_CEncoder(n_nodes, n_hidden, n_embedding,
                                   n_nodes, keepprob=1, use_cuda=False)
        self.decoder = Fc_CDecoder(n_nodes, n_hidden, n_embedding,
                                   n_nodes, keepprob=1, use_cuda=False)
        if self.use_cuda:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()

        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)

        if loss is None:
            self.loss = nn.BCELoss(reduction="sum")
        else:
            self.loss = loss

        self.epoch = 0
        self.criterion = np.inf

        
        if optimizer is None:
            self.optimizer = torch.optim.SGD(self.parameters(), lr = 1e-2)
        else:
            self.optimizer = optimizer(self.parameters())



    def _sample_embedding(self, mu, var):
        var = 1e-6 + F.softplus(var)
        batch_size = mu.size(0)
        eps = torch.randn(batch_size, self.n_embedding)

        if self.use_cuda:
            eps = eps.cuda()

        return mu + var * eps


    def _vae_loss(self, inputs, outputs):
        recon_data = outputs[0]
        mu = outputs[1]
        var= outputs[2]

        batch_size = inputs.size(0)
        recon_loss = self.loss(recon_data, inputs) / batch_size
        
        KL_loss = 0.5 * torch.sum(
                                torch.pow(mu, 2) +
                                torch.pow(var, 2) -
                                torch.log(1e-8 + torch.pow(var, 2)) - 1
                               ).sum() / batch_size
        return recon_loss + KL_loss


    def forward(self, x, c):
        z_mu, z_var = self.encoder(x, c)
        z = self._sample_embedding(z_mu, z_var)
        y = self.decoder(z, c)
        
        return y, z_mu, z_var
    

    def predict(self, past_states, batch_size=32):
        if past_states.dim() == 1:
            past_states = past_states.repeat(batch_size, 1)

        if self.use_cuda:
            past_states = past_states.cuda()

        self.train(False)
        self.eval()
        z = torch.randn(batch_size, self.n_embedding)
        if self.use_cuda:
            z = z.cuda()
        sample = self.decoder(z, past_states).detach().cpu().data.numpy()
        self.train(True)
        
        return sample, z, past_states


    def evaluate(self, dataset, batch_size=64):
        
        self.train(False)
        self.eval()
        loss_value = []

        data_loader = DataLoader(dataset, batch_size)

        for batch in data_loader:
        
            inputs, past_states = batch
            
            if self.use_cuda:
                inputs = inputs.cuda()
                past_states = past_states.cuda()

            outputs = self.forward(inputs, past_states)
            loss = self._vae_loss(inputs, outputs)
            loss_value.append(loss.data)
        self.train(True)

        return sum(loss_value) / len(loss_value)


    def fit(self, train_dataset, val_dataset=None, epochs=10, batch_size=64,
            verbose=True, keep_best=True):

        train_loader = DataLoader(train_dataset, batch_size)

        self.train(True)

        start = time.time()
        for i in range(epochs):

            for j, batch in enumerate(train_loader):
                self.optimizer.zero_grad()
                inputs, past_states = batch
            
                if self.use_cuda:
                    inputs = inputs.cuda()
                    past_states = past_states.cuda()

                outputs = self.forward(inputs, past_states)
                loss = self._vae_loss(inputs, outputs)
                loss.backward()
                self.optimizer.step()

            train_loss = self.evaluate(train_dataset, batch_size)
            criterion = train_loss
            if val_dataset is not None:
                val_loss = self.evaluate(val_dataset, batch_size)
                criterion = val_loss
            else:
                val_loss = 0

            if criterion < self.criterion:
                self.criterion = criterion
                if keep_best:
                    best_param = self.state_dict()
                    new_best = True
            else:
                new_best = False


            if verbose:
                end = time.time()

                if new_best:
                    print(f"Epoch {self.epoch} - " +\
                          f"Training Loss: {avg_train_loss:0.4f} ± " +\
                          f"{std_train_loss:0.4f} - " +\
                          f"Validation Loss: {avg_val_loss:0.4f} ± " +\
                          f"{std_val_loss:0.4f} - " +\
                          f"Training time: {end - start:0.04f} - " +\
                          f"New best config.")
                else:
                    print(f"Epoch {self.epoch} - " +\
                          f"Training Loss: {train_loss:0.4f} ± " +\
                          f"{std_train_loss:0.4f} - " +\
                          f"Validation Loss: {val_loss:0.4f} ± " +\
                          f"{std_val_loss:0.4f} - " +\
                          f"Training time: {end - start:0.04f} - ")
                start = time.time()
            self.epoch += 1

        if keep_best: self.load_state_dict(best_param)
        return None


    def save_params(self, f):
        torch.save(self.state_dict(), f)


    def load_params(self, f):
        params = torch.load(f, map_location='cpu')
        self.load_state_dict(params)


    def save_optimizer(self, f):
        """
        Saves the state of the current optimizer.

        Args:
            f: File-like object (has to implement fileno that returns a file
                descriptor) or string containing a file name.
        """
        torch.save(self.optimizer.state_dict(), f)


    def load_optimizer(self, f):
        """
        Loads the optimizer state saved using the ``torch.save()`` method or the
        ``save_optimizer_state()`` method of this class.

        Args:
            f: File-like object (has to implement fileno that returns a file
                descriptor) or string containing a file name.
        """
        self.optimizer.load_state_dict(torch.load(f, map_location='cpu'))


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)