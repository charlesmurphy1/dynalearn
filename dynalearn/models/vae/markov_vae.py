import numpy as np
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F

from .history import History


class MarkovVAE(nn.Module):
    def __init__(self):
        super(MarkovVAE, self).__init__()
        

        self.encoder = None
        self.decoder = None
        self.optimizer = None
        self.scheduler = None
        self.loss = None

        self.use_cuda = False
        self.epoch = 0
        self.criterion = np.inf

        self.history = History("history")
        self.current_param = None

    def _get_embedding_size(self):
        raise NotImplemented("self._get_embedding_size() has not been" +\
                             "implemented.")


    def _get_past_size(self):
        raise NotImplemented("self._get_past_size() has not been" +\
                             "implemented.")


    def _get_conditional(self, past):
        raise NotImplemented("self._get_conditional() has not been" +\
                             "implemented.")


    def setup_trainer(self, optimizer, loss, scheduler):
        if optimizer is None:
            self.optimizer = torch.optim.SGD(self.parameters(), lr = 1e-3)
        else:
            self.optimizer = optimizer(self.parameters())

        if loss is None:
            self.loss = nn.BCELoss(reduction="sum")
        else:
            self.loss = loss

        if scheduler is None:
            f = lambda epoch: 1
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, f)
        else:
            self.scheduler = scheduler(self.optimizer)

    def _model_loss(self, states, outputs, beta=1):
        recon_states = outputs[0]
        mu = outputs[1]
        var= outputs[2]

        batch_size = states.size(0)
        recon_loss = self.loss(recon_states, states)
        
        KL_loss = 0.5 * (torch.pow(mu, 2) + torch.pow(var, 2) -\
                  torch.log(1e-8 + torch.pow(var, 2)) - 1)

        return torch.mean(recon_loss), beta * torch.mean(KL_loss)

    def _sample_embedding(self, mu, var):
        var = 1e-6 + F.softplus(var)
        batch_size = mu.size(0)
        eps = torch.randn(batch_size, *self._get_embedding_size())

        if self.use_cuda:
            eps = eps.cuda()
        return mu + var * eps

    def forward(self, present, past):
        conditional = self._get_conditional(past)

        z_mu, z_var = self.encoder(present, conditional)
        z = self._sample_embedding(z_mu, z_var)
        y = self.decoder(z, conditional)
        
        return y, z_mu, z_var


    def predict(self, past, batch_size=32):
        self.train(False)
        self.eval()

        if past.dim() < 1 + len(self._get_past_size()):
            s = [batch_size, *[1] * len(self._get_past_size())]
            past = past.view(1, *self._get_past_size()).repeat(*s)
        elif past.size(0) != batch_size:
            raise ValueError("Size at dimension 0 does not match batch_size.")
        else:
            past = past.clone()

        conditional = self._get_conditional(past)

        z = torch.randn(batch_size, *self._get_embedding_size())

        if self.use_cuda:
            z = z.cuda()
            conditional = conditional.cuda()

        sample = self.decoder(z, conditional).detach().cpu().numpy()
        self.train(True)

        return sample, z, past


    def evaluate_on_dataset(self, dataset,
                            training_metrics=None,
                            batch_size=64,
                            beta=1):
        
        self.train(False)
        self.eval()

        data_loader = DataLoader(dataset, batch_size)
        n = len(data_loader)
        measures = {}
        if training_metrics:
            for m in training_metrics:
                measures[m] = np.zeros(n)
        else:
            training_metrics = []
        for i, batch in enumerate(data_loader):
            present, past = batch
            if self.use_cuda:
                present = present.cuda()
                past = past.cuda()

            outputs = self.forward(present, past)
            recon, kl_div = self._model_loss(present, outputs, beta)
            recon = recon.detach().cpu().numpy()
            kl_div = kl_div.detach().cpu().numpy()

            for j, m in enumerate(training_metrics):
                if type(m) is tuple:
                    measures[m][i] = training_metrics[j](present, outputs).detach().cpu().numpy()
                elif m == "loss":
                    measures[m][i] = recon + kl_div
                elif m == "recon":
                    measures[m][i] = recon
                elif m == "kl_div":
                    measures[m][i] = kl_div

        for m in measures:
            measures[m] = (np.mean(measures[m]), np.std(measures[m]))

        self.train(True)

        return measures

    def evaluate_on_model(self,
                          model_metrics=None,
                          batch_size=64):

        self.train(False)
        self.eval()

        measures = {}
        if model_metrics is None:
            model_metrics = []

        for j, m in enumerate(model_metrics):
            if m == "learning_rate" or m == "lr":
                for g in self.optimizer.param_groups:
                    measures[m] = (g['lr'], 0)
            if m == "weight_decay" or m == "wd":
                for g in self.optimizer.param_groups:
                    measures[m] = (g['weight_decay'], 0)
            if m == "param":
                params = []
                for name, data in self.named_parameters():
                    p = data.detach().cpu().numpy()
                    params.append(np.mean(p), np.std(p)**2)
                params = np.array(params)
                measures[m] = (np.mean(params, 0), np.sqrt(np.mean(params, 1)))
            if m == "grad":
                params = []
                for name, data in self.named_parameters():
                    g = data.grad.detach().cpu().numpy()
                    params.append(np.mean(g), np.std(g)**2)
                params = np.array(params)
                measures[m] = (np.mean(params, 0), np.sqrt(np.mean(params, 1)))

        self.train(True)

        return measures




    def fit(self, train_dataset, val_dataset=None, epochs=10, batch_size=64,
            verbose=True, keep_best=True, training_metrics=None, 
            model_metrics=None, show_var=False, beta=1):

        train_loader = DataLoader(train_dataset, batch_size)

        self.train(True)

        if self.current_param:
            self.load_state_dict(self.current_param)
        else:
            self.current_param = self.state_dict()
            self.best_param = self.state_dict()

        start = time.time()

        # Evaluating model before starting the training

        # Show initial progression
        if self.epoch == 0:
            train_measures = self.evaluate_on_dataset(train_dataset,
                                          training_metrics=training_metrics,
                                          batch_size=batch_size,
                                          beta=beta)
            if val_dataset is not None:
                val_measures = self.evaluate_on_dataset(val_dataset,
                                            training_metrics=training_metrics,
                                            batch_size=batch_size,
                                            beta=beta)
            else:
                val_measures = {}

            model_measures = {}

            self.history.evaluate_metrics(self.epoch,
                                          train_measures,
                                          val_measures,
                                          model_measures)
            if verbose:
                self.history.progression(self.epoch, 0,
                                         training_metrics=training_metrics,
                                         is_best=True,
                                         show_var=show_var)

        # Start training
        for i in range(epochs):
            self.epoch += 1
            for j, batch in enumerate(train_loader):

                self.optimizer.zero_grad()
                present, past = batch

                if self.use_cuda:
                    present = present.cuda()
                    past = past.cuda()

                outputs = self.forward(present, past)

                recon, KL = self._model_loss(present, outputs, beta)
                loss = recon + KL
                loss.backward()
                self.optimizer.step()

            # Evaluating metrics
            train_measures = self.evaluate_on_dataset(train_dataset,
                                           training_metrics=training_metrics,
                                           batch_size=batch_size,
                                           beta=beta)

            if val_dataset is not None:
                val_measures = self.evaluate_on_dataset(val_dataset,
                                             training_metrics=training_metrics,
                                             batch_size=batch_size,
                                             beta=beta)
            else:
                val_measures = {}

            model_measures = self.evaluate_on_model(model_metrics=model_metrics,
                                                    batch_size=batch_size)

            new_criterion = self.history.evaluate_metrics(self.epoch,
                                                          train_measures,
                                                          val_measures,
                                                          model_measures)

            # Checking for best configurations
            if new_criterion <= self.criterion:
                self.criterion = new_criterion
                if keep_best:
                    self.best_param = self.state_dict()
                    new_best = True
            else:
                new_best = False


            if verbose:
                end = time.time()
                self.history.progression(self.epoch,
                                         end - start,
                                         training_metrics=training_metrics,
                                         is_best=new_best,
                                         show_var=show_var)
                start = time.time()

        self.current_param = self.state_dict()
        if keep_best: self.load_state_dict(self.best_param)
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
