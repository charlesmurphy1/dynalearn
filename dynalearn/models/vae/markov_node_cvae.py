import numpy as np
import math
from random import randint
import sys
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.init as init
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .fc_cvae import Fc_CEncoder, Fc_CDecoder
from .cvae import CVAE

import time

class NodeLinear(nn.Module):
    def __init__(self, num_nodes, in_features, out_features, bias=True):
        super(NodeLinear, self).__init__()
        self.num_nodes = num_nodes
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.Tensor(num_nodes,
                                             in_features,
                                             out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(num_nodes,
                                               out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        batch_size = input.size(0)
        input = input.view(batch_size, self.num_nodes, 1, self.in_features)
        ans = torch.matmul(input, self.weight).view(batch_size,
                                                    self.num_nodes,
                                                    self.out_features)
        if self.bias is not None:
            return ans + self.bias
        else:
            return ans

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class Markov_Node_CVAE(nn.Module):
    def __init__(self, graph, n_hidden, n_embedding,
                 optimizer=None, loss=None, metrics=None, 
                 use_cuda=False, keepprob=1):
        super(Markov_Node_CVAE, self).__init__()
        
        self.graph = graph
        self.num_nodes = self.graph.number_of_nodes()
        self.degrees = dict(self.graph.degree())
        self.n_hidden = n_hidden
        self.n_embedding = n_embedding
        self.use_cuda = use_cuda

        relu = nn.ReLU()
        tanh = nn.Tanh()
        sigmoid = nn.Sigmoid()
        dropout = nn.Dropout(1 - keepprob)

        # Encoder
        layers = [NodeLinear(self.num_nodes,
                             self.num_nodes + 1,
                             self.n_hidden[0])]

        for i in range(1, len(n_hidden)):
            layers.append(relu)
            layers.append(dropout)
            layers.append(NodeLinear(self.num_nodes,
                                     self.n_hidden[i - 1], 
                                     self.n_hidden[i]))
        self.encoder = nn.Sequential(*layers, tanh, dropout)
        self.mu = NodeLinear(self.num_nodes,
                             self.n_hidden[-1],
                             self.n_embedding)
        self.var = NodeLinear(self.num_nodes,
                              self.n_hidden[-1],
                              self.n_embedding)

        # Decoder
        layers = []
        for i in range(len(n_hidden) - 1, 0, -1):
            layers.append(NodeLinear(self.num_nodes, 
                                     self.n_hidden[i],
                                     self.n_hidden[i - 1]))
            layers.append(relu)
            layers.append(dropout)
        self.decoder = nn.Sequential(NodeLinear(self.num_nodes,
                                                self.n_embedding + self.num_nodes,
                                                self.n_hidden[-1]),
                                     tanh,
                                     dropout,
                                     *layers,
                                     NodeLinear(self.num_nodes,
                                                self.n_hidden[0],
                                                1),
                                     sigmoid)

        if self.use_cuda:
            self.encoder = self.encoder.cuda()
            self.mu = self.mu.cuda()
            self.var = self.var.cuda()
            self.decoder = self.decoder.cuda()

        self.encoder.apply(init_weights)
        self.mu.apply(init_weights)
        self.var.apply(init_weights)
        self.decoder.apply(init_weights)

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
        eps = torch.randn(batch_size, self.num_nodes, self.n_embedding)

        if self.use_cuda:
            eps = eps.cuda()

        return mu + var * eps


    def _vae_loss(self, states, outputs):
        recon_states = outputs[0]
        mu = outputs[1]
        var= outputs[2]

        batch_size = states.size(0)
        recon_loss = self.loss(recon_states, states) / batch_size
        
        KL_loss = 0.5 * torch.sum(
                                    torch.pow(mu, 2) +
                                    torch.pow(var, 2) -
                                    torch.log(1e-8 + torch.pow(var, 2)) - 1
                                 ) / batch_size

        return recon_loss, KL_loss


    def forward(self, x):
        present, past = x[0], x[1]
        batch_size = present.size(0)
        conditional = past.repeat(1, 1, self.num_nodes)

        x = torch.cat([present, conditional], 2)

        h = self.encoder(x)
        z_mu, z_var = self.mu(h), self.var(h)
        z = self._sample_embedding(z_mu, z_var)
        z = torch.cat([z, conditional], 2)

        y = self.decoder(z)
        
        return y, z_mu, z_var
    

    def predict(self, past, batch_size=32):
        # print(past.size())

        if past.dim() < 3:
            _past = past.view(1, self.num_nodes, 1).repeat(batch_size, 1, 1)
        else:
            _past = past.clone()

        conditional = _past.repeat(1, 1, self.num_nodes)
        sample = {}
        z = {}
        self.train(False)
        self.eval()
        
        z = torch.randn(batch_size, self.num_nodes, self.n_embedding)
        z = torch.cat([z, conditional], 2)

        sample = self.decoder(z).detach().cpu().data.numpy()

        self.train(True)

        return sample, z, past


    def evaluate(self, dataset, batch_size=64):
        
        self.train(False)
        self.eval()

        data_loader = DataLoader(dataset, batch_size)
        loss_value = np.zeros(len(data_loader))
        recon_value = np.zeros(len(data_loader))
        kl_value = np.zeros(len(data_loader))
        i = 0

        for batch in data_loader:
            present, past = batch
            if self.use_cuda:
                present = present.cuda()
                past = past.cuda()

            outputs = self.forward((present, past))
            recon, KL = self._vae_loss(present,
                                       outputs
                                      )
            loss_value[i] = recon.detach().cpu().numpy() +\
                            KL.detach().cpu().numpy()
            recon_value[i] = recon.detach().cpu().numpy()
            kl_value[i] = KL.detach().cpu().numpy()
            i += 1

        self.train(True)
        return (np.mean(loss_value), np.std(loss_value)), \
               (np.mean(recon_value), np.std(recon_value)), \
               (np.mean(kl_value), np.std(kl_value))


    def fit(self, train_dataset, val_dataset=None, epochs=10, batch_size=64,
            verbose=True, keep_best=True):

        train_loader = DataLoader(train_dataset, batch_size)

        self.train(True)
        start = time.time()
        train_metrics = self.evaluate(train_dataset,
                                        batch_size)
        if val_dataset is not None:
            val_metrics = self.evaluate(val_dataset,
                                        batch_size)
        else:
            val_metrics = None
        if verbose and self.epoch == 0:
            self.progress(self.epoch,
                          0,
                          train_metrics,
                          val_metrics,
                          False)
        for i in range(epochs):

            self.epoch += 1
            for j, batch in enumerate(train_loader):

                self.optimizer.zero_grad()
                present, past = batch

                if self.use_cuda:
                    present = present.cuda()
                    past = past.cuda()

                outputs = self.forward((present, past))

                recon, KL = self._vae_loss(present, outputs)
                loss = recon + KL
                loss.backward()
                self.optimizer.step()

            train_metrics = self.evaluate(train_dataset,
                                          batch_size)
            new_criterion = train_metrics[0][0]
            if val_dataset is not None:
                val_metrics = self.evaluate(val_dataset,
                                            batch_size)
                new_criterion = val_metrics[0][0]
            else:
                val_metrics = None

            if new_criterion < self.criterion:
                self.criterion = new_criterion
                if keep_best:
                    self.best_param = self.state_dict()
                    new_best = True
            else:
                new_best = False


            if verbose:
                end = time.time()

                self.progress(self.epoch,
                              end - start,
                              train_metrics,
                              val_metrics,
                              new_best)

                start = time.time()

        if keep_best: self.load_state_dict(self.best_param)
        return None


    def progress(self, epoch, time, train_metrics,
                 val_metrics=None, is_best=False):
        if is_best: sys.stdout.write(f"New best epoch: {epoch} "+\
                                     f"- Time: {time:0.02f}\n")
        else: sys.stdout.write(f"Epoch: {epoch} - Time: {time:0.02f}\n")

        loss, recon, kl = train_metrics
        sys.stdout.write(f"\t Training - " +\
                         f"Loss: {loss[0]:0.4f} ± {loss[1]:0.2f}, " +\
                         f"Recon.: {recon[0]:0.4f} ± {recon[1]:0.2f}, " +\
                         f"KL-div.: {kl[0]:0.4f} ± {kl[1]:0.2f}\n")
        if val_metrics:
            loss, recon, kl = val_metrics
            sys.stdout.write(f"\t Validation - " +\
                             f"Loss: {loss[0]:0.4f} ± {loss[1]:0.2f}, " +\
                             f"Recon.: {recon[0]:0.4f} ± {recon[1]:0.2f}, " +\
                             f"KL-div.: {kl[0]:0.4f} ± {kl[1]:0.2f}\n")
        sys.stdout.flush()


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