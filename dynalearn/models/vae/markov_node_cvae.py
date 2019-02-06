import numpy as np
from random import randint
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
from .fc_cvae import Fc_CEncoder, Fc_CDecoder
from .cvae import CVAE
from .markov_layers import NodeEncoder, NodeDecoder, SparseNodeEncoder

import time


class Markov_Node_CVAE(nn.Module):
    def __init__(self, encoder, decoder, optimizer=None,
                 loss=None, use_cuda=False):
        super(Markov_Node_CVAE, self).__init__()
        

        self.encoder = encoder
        self.decoder = decoder

        self.num_nodes = self.encoder.num_nodes
        self.n_embedding = self.encoder.n_embedding
        self.use_cuda = use_cuda

        if self.use_cuda:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()

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


    def _model_loss(self, states, outputs):
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
        x, past = x[0], x[1]
        batch_size = x.size(0)
        conditional = past.repeat(1, 1, self.num_nodes)

        z_mu, z_var = self.encoder((x, conditional))
        z = self._sample_embedding(z_mu, z_var)
        y = self.decoder((z, conditional))
        
        return y, z_mu, z_var
    

    def predict(self, past, batch_size=32):
        self.train(False)
        self.eval()

        if past.dim() < 3:
            _past = past.view(1, self.num_nodes, 1).repeat(batch_size, 1, 1)
        else:
            _past = past.clone()
        conditional = _past.repeat(1, 1, self.num_nodes)

        z = torch.randn(batch_size, self.num_nodes, self.n_embedding)
        if self.use_cuda:
            z = z.cuda()
            conditional = conditional.cuda()
        sample = self.decoder((z, conditional)).detach().cpu().numpy()
        self.train(True)

        return sample, z, past


    def evaluate(self, dataset, metrics=None, batch_size=64):
        
        self.train(False)
        self.eval()

        data_loader = DataLoader(dataset, batch_size)
        n = len(data_loader)
        measures = {'loss': np.zeros(n)}
        if metrics:
            for m in metrics:
                measures[m] = np.zeros(n)
        for i, batch in enumerate(data_loader):
            present, past = batch
            if self.use_cuda:
                present = present.cuda()
                past = past.cuda()

            outputs = self.forward((present, past))
            recon, kl_div = self._model_loss(present,
                                             outputs
                                        )
            recon = recon.detach().cpu().numpy()
            kl_div = kl_div.detach().cpu().numpy()

            measures["loss"][i] = recon + kl_div

            for j, m in enumerate(metrics):
                if type(m) is tuple:
                    measures[m][i] = metrics[j](present, outputs).detach().cpu().numpy()
                elif m == "recon":
                    measures[m][i] = recon
                elif m == "kl_div":
                    measures[m][i] = kl_div

        for m in measures:
            measures[m] = (np.mean(measures[m]), np.std(measures[m]))

        self.train(True)
        return measures


    def fit(self, train_dataset, val_dataset=None, epochs=10, batch_size=64,
            verbose=True, keep_best=True, metrics=None, show_var=False):

        train_loader = DataLoader(train_dataset, batch_size)

        self.train(True)
        start = time.time()
        train_metrics = self.evaluate(train_dataset,
                                      metrics=metrics,
                                      batch_size=batch_size)
        if val_dataset is not None:
            val_metrics = self.evaluate(val_dataset,
                                        metrics=metrics,
                                        batch_size=batch_size)
        else:
            val_metrics = None
        if verbose and self.epoch == 0:
            self.progress(self.epoch,
                          0,
                          train_metrics,
                          val_metrics,
                          False,
                          show_var)
        for i in range(epochs):

            self.epoch += 1
            for j, batch in enumerate(train_loader):

                self.optimizer.zero_grad()
                present, past = batch

                if self.use_cuda:
                    present = present.cuda()
                    past = past.cuda()

                outputs = self.forward((present, past))

                recon, KL = self._model_loss(present, outputs)
                loss = recon + KL
                loss.backward()
                self.optimizer.step()

            train_metrics = self.evaluate(train_dataset,
                                          metrics=metrics,
                                          batch_size=batch_size)
            new_criterion = train_metrics["loss"][0]
            if val_dataset is not None:
                val_metrics = self.evaluate(val_dataset,
                                          metrics=metrics,
                                          batch_size=batch_size)
                new_criterion = val_metrics["loss"][0]
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
                              new_best,
                              show_var)
                start = time.time()

        if keep_best: self.load_state_dict(self.best_param)
        return None


    def progress(self, epoch, time, train_metrics,
                 val_metrics=None, is_best=False,
                 show_var=True):
        if is_best: sys.stdout.write(f"New best Epoch: {epoch} "+\
                                     f"- Time: {time:0.02f}\n")
        else: sys.stdout.write(f"Epoch: {epoch} - Time: {time:0.02f}\n")

        loss, recon, kl = train_metrics
        sys.stdout.write(f"\t Train. - ")
        for m in train_metrics:
            sys.stdout.write(f"{m}: {train_metrics[m][0]:0.4f}")
            if show_var:
                sys.stdout.write(f" ± {train_metrics[m][1]:0.2f}")
            sys.stdout.write(f", ")
        sys.stdout.write(f"\n")

        if val_metrics:
            sys.stdout.write(f"\t Val. - ")
            for m in val_metrics:
                sys.stdout.write(f"{m}: {val_metrics[m][0]:0.4f}")
                if show_var:
                    sys.stdout.write(f" ± {val_metrics[m][1]:0.2f}")
                sys.stdout.write(f", ")
            sys.stdout.write(f"\n")
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



def basicMarkovCVAE(graph, n_hidden, n_embedding,
                    keepprob=1, optimizer=None,
                    loss=None, use_cuda=False):
    
    encoder = NodeEncoder(graph, n_hidden, n_embedding, keepprob)
    decoder = NodeDecoder(graph, n_hidden, n_embedding, keepprob)

    return Markov_Node_CVAE(encoder, decoder, optimizer, loss, use_cuda)


def sparseMarkovCVAE(graph, n_hidden, n_embedding,
                    keepprob=1, optimizer=None,
                    loss=None, use_cuda=False):
    encoder = SparseNodeEncoder(graph, n_hidden, n_embedding, keepprob)
    decoder = NodeDecoder(graph, n_hidden, n_embedding, keepprob)

    return Markov_Node_CVAE(encoder, decoder, optimizer, loss, use_cuda)