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


class Markov_Node_CVAE(nn.Module):
    def __init__(self, graph, n_hidden, n_embedding,
                 optimizer=None, loss=None, use_cuda=False):
        super(Markov_Node_CVAE, self).__init__()
        
        self.graph = graph
        self.n_nodes = self.graph.number_of_nodes()
        self.degrees = dict(self.graph.degree())
        self.n_hidden = n_hidden
        self.n_embedding = n_embedding
        self.use_cuda = use_cuda
        # self.encoders = {}
        # self.decoders = {}

        # for n in self.graph.node:
        #     self.encoders[str(n)] = Fc_CEncoder(1, n_hidden, n_embedding,
        #                                         self.degrees[n] + 1,
        #                                         keepprob=1,
        #                                         use_cuda=self.use_cuda)
        #     self.decoders[str(n)] = Fc_CDecoder(1, n_hidden, n_embedding,
        #                                         self.degrees[n] + 1,
        #                                         keepprob=1,
        #                                         use_cuda=self.use_cuda)
        #     if self.use_cuda:
        #         self.encoders[str(n)] = self.encoders[str(n)].cuda()
        #         self.decoders[str(n)] = self.decoders[str(n)].cuda()

        # self.encoders = nn.ModuleDict(self.encoders)
        # self.decoders = nn.ModuleDict(self.decoders)
        self.encoders.apply(init_weights)
        self.decoders.apply(init_weights)

        if optimizer is None:
            self.optimizer = torch.optim.SGD(self.parameters(), lr = 1e-2)
        else:
            self.optimizer = optimizer(self.parameters())


        if loss is None:
            self.loss = nn.BCELoss(reduction="sum")
        else:
            self.loss = loss

        self.epoch = 0
        self.criterion = np.inf

        



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


    def forward(self, node, x, c):
        z_mu, z_var = self.encoders[str(node)](x, c)
        z = self._sample_embedding(z_mu, z_var)
        y = self.decoders[str(node)](z, c)
        
        return y, z_mu, z_var
    

    def predict(self, past_states, batch_size=32):

        sample = {}
        z = {}
        self.train(False)
        self.eval()
        # for n in self.graph.node:
        #     if past_states[n].dim() == 1:
        #         past_states[n] = past_states[n].repeat(batch_size, 1)

        #     if self.use_cuda:
        #         past_states[n] = past_states[n].cuda()


        #     z[n] = torch.randn(batch_size, self.n_embedding)
        #     if self.use_cuda:
        #         z[n] = z[n].cuda()
        #     sample[n] = self.decoders[str(n)](z[n],
        #                                       past_states[n]
        #                                      ).detach().cpu().data.numpy()
        self.train(True)

        return sample, z, past_states


    def evaluate(self, dataset, batch_size=64):
        
        self.train(False)
        self.eval()

        data_loader = DataLoader(dataset, batch_size)
        loss_value = np.zeros(len(data_loader) * self.n_nodes)
        i = 0

        for batch in data_loader:
            inputs, past_states = batch
            # for n in self.graph.node:
            #     node_input = inputs[n]
            #     node_past = past_states[n]
            #     if self.use_cuda:
            #         node_input = node_input.cuda()
            #         node_past = node_past.cuda()

            #     node_output = self.forward(n, node_input, node_past)
            #     loss_value[i] = self._vae_loss(node_input, node_output).clone().cpu().detach().numpy()
            #     i += 1
        self.train(True)

        return np.mean(loss_value), np.std(loss_value)


    def fit(self, train_dataset, val_dataset=None, epochs=10, batch_size=64,
            verbose=True, keep_best=True):

        train_loader = DataLoader(train_dataset, batch_size)

        self.train(True)

        start = time.time()
        for i in range(epochs):

            for j, batch in enumerate(train_loader):
                loss = 0    
                inputs, past_states = batch
                self.optimizer.zero_grad()
                # for n in self.graph.node:

                #     node_input = inputs[n]
                #     node_past = past_states[n]
                #     if self.use_cuda:
                #         node_input = node_input.cuda()
                #         node_past = node_past.cuda()
                
                #     node_output = self.forward(n, node_input, node_past)
                #     loss += self._vae_loss(node_input, node_output)

                loss.backward()
                self.optimizer.step()

            avg_train_loss, std_train_loss = self.evaluate(train_dataset,
                                                           batch_size)
            new_criterion = avg_train_loss
            if val_dataset is not None:
                avg_val_loss, std_val_loss = self.evaluate(val_dataset,
                                                           batch_size)
                new_criterion = avg_val_loss
            else:
                avg_val_loss = 0

            if new_criterion < self.criterion:
                self.criterion = new_criterion
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
                          f"{std_train_loss:0.2f} - " +\
                          f"Validation Loss: {avg_val_loss:0.4f} ± " +\
                          f"{std_val_loss:0.2f} - " +\
                          f"Training time: {end - start:0.04f} - " +\
                          f"New best config.")
                else:
                    print(f"Epoch {self.epoch} - " +\
                          f"Training Loss: {avg_train_loss:0.4f} ± " +\
                          f"{std_train_loss:0.2f} - " +\
                          f"Validation Loss: {avg_val_loss:0.4f} ± " +\
                          f"{std_val_loss:0.2f} - " +\
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