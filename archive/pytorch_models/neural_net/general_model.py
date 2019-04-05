import numpy as np
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F

from dynalearn.pytorch_models.history import History


class MarkovNN(nn.Module):
    def __init__(self):
        super(MarkovNN, self).__init__()
        

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss = None

        self.use_cuda = False
        self.epoch = 0
        self.criterion = np.inf

        self.history = History("history")
        self.current_param = None


    def compile(self, optimizer=None, loss=None, scheduler=None):
        if optimizer is None:
            self.optimizer = torch.optim.SGD(self.parameters(), lr = 1e-3)
        else:
            self.optimizer = optimizer(self.parameters())

        if loss is None:
            self.loss = nn.CrossEntropy(reduction="none")
        else:
            self.loss = loss

        if scheduler is None:
            f = lambda epoch: 1
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, f)
        else:
            self.scheduler = scheduler(self.optimizer)


    def forward(self, inputs, adj):

        outputs = self.model(inputs, adj)
        return outputs


    def predict(self, inputs, adj, batch_size=32):

        self.train(False)
        self.eval()
        outputs = self.forward(inputs, adj).detach().cpu()
        self.train(True)

        return outputs


    def evaluate_on_dataset(self, dataset,
                            training_metrics=None,
                            batch_size=64):
        
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
            inputs, adj, targets = batch
            if self.use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()

            outputs = self.forward(inputs, adj)
            loss = self.loss(outputs, targets).mean().detach().cpu().numpy()
            loss = np.sum(loss)

            for j, m in enumerate(training_metrics):
                if type(m) is tuple:
                    measures[m][i] = training_metrics[j](outputs, targets).detach().cpu().numpy()
                elif m == "loss":
                    measures[m][i] = loss

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
            verbose=True, progress_bar=None, keep_best=True,
            training_metrics=None,  model_metrics=None,
            show_var=False):

        train_loader = DataLoader(train_dataset, batch_size)

        self.train(True)

        if self.current_param:
            self.load_state_dict(self.current_param)
        else:
            self.current_param = self.state_dict()
            self.best_param = self.state_dict()

        start = time.time()

        # Evaluating model before starting the training
        ## Show initial progression
        if self.epoch == 0:
            train_measures = self.evaluate_on_dataset(train_dataset,
                                          training_metrics=training_metrics,
                                          batch_size=batch_size)
            if val_dataset is not None:
                val_measures = self.evaluate_on_dataset(val_dataset,
                                            training_metrics=training_metrics,
                                            batch_size=batch_size)
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

            if progress_bar:
                bar = progress_bar(range(len(train_loader)), 'Epoch {}'.format(i))

            for j, batch in enumerate(train_loader):

                self.optimizer.zero_grad()
                inputs, adj, targets = batch

                if self.use_cuda:
                    inputs = inputs.cuda()
                    adj = adj.cuda()
                    targets = targets.cuda()

                outputs = self.forward(inputs, adj)

                loss = -self.loss(outputs, targets).mean()

                loss.backward()
                self.optimizer.step()
                if progress_bar:
                    bar.update()

            if progress_bar:
                bar.close()



            # Evaluating metrics
            train_measures = self.evaluate_on_dataset(train_dataset,
                                           training_metrics=training_metrics,
                                           batch_size=batch_size)

            if val_dataset is not None:
                val_measures = self.evaluate_on_dataset(val_dataset,
                                             training_metrics=training_metrics,
                                             batch_size=batch_size)
            else:
                val_measures = {}

            model_measures = self.evaluate_on_model(model_metrics=model_metrics,
                                                    batch_size=batch_size)

            new_criterion = self.history.evaluate_metrics(self.epoch,
                                                          train_measures,
                                                          val_measures,
                                                          model_measures)

            if type(self.scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                self.scheduler.step(new_criterion)
            else:
                self.scheduler.step()

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


    def save_params(self, path):
        with open(path, "wb") as f:
            torch.save(self.state_dict(), f)


    def load_params(self, path):
        with open(path, "rb") as f:
            params = torch.load(f, map_location='cpu')
        self.load_state_dict(params)


    def save_optimizer(self, path):
        """
        Saves the state of the current optimizer.

        Args:
            f: File-like object (has to implement fileno that returns a file
                descriptor) or string containing a file name.
        """
        with open(path, "wb") as f:
            torch.save(self.optimizer.state_dict(), f)


    def load_optimizer(self, path):
        """
        Loads the optimizer state saved using the ``torch.save()`` method or the
        ``save_optimizer_state()`` method of this class.

        Args:
            f: File-like object (has to implement fileno that returns a file
                descriptor) or string containing a file name.
        """
        with open(path, "rb") as f:
            self.optimizer.load_state_dict(torch.load(f, map_location='cpu'))


    def save_history(self, path):
        self.history.save(path)


    def load_history(self, path):
        self.history.load(path)


    def save_state(self, path):
        import os

        if not os.path.exists(path):
            os.mkdir(path)
            
        self.save_params(os.path.join(path, "params.pt"))
        self.save_optimizer(os.path.join(path, "optimizer.pt"))
        self.save_history(os.path.join(path, "history.pt"))


