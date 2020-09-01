import networkx as nx
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import tqdm

from abc import abstractmethod
from dynalearn.config import Config
from dynalearn.nn.callbacks import CallbackList
from dynalearn.nn.history import History
from dynalearn.nn.optimizer import get as get_optimizer
from dynalearn.utilities import to_edge_index


class GraphNeuralNetwork(torch.nn.Module):
    def __init__(self, config=None, **kwargs):
        torch.nn.Module.__init__(self)
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        self.get_optimizer = get_optimizer(config.optimizer)
        self.history = History()
        if "using_log" in config.__dict__:
            self.using_log = config.using_log
        else:
            self.using_log = False

    @abstractmethod
    def forward(self, x, edge_index):
        raise NotImplemented()

    @abstractmethod
    def loss(self, y_true, y_pred, weights):
        raise NotImplemented()

    def fit(
        self,
        dataset,
        epochs=1,
        batch_size=1,
        learning_rate=1e-3,
        val_dataset=None,
        metrics={},
        callbacks=[],
        verbose=0,
    ):
        if type(callbacks) == list:
            callbacks = CallbackList(callbacks)
            for c in callbacks:
                if "verbose" in c.__dict__:
                    c.verbose = verbose > 0

        callbacks.set_params(self)
        callbacks.set_model(self)
        callbacks.on_train_begin()
        self.setUp(dataset)
        for i in range(epochs):
            callbacks.on_epoch_begin(self.history.epoch)
            t0 = time.time()
            self._do_epoch_(
                dataset, batch_size=batch_size, callbacks=callbacks, verbose=verbose
            )

            train_metrics = self.evaluate(dataset, metrics)
            if val_dataset is not None:
                val_metrics = self.evaluate(val_dataset, metrics, "val")
            else:
                val_metrics = {}

            t1 = time.time()
            logs = {"epoch": self.history.epoch + 1, "time": t1 - t0}
            logs.update(train_metrics)
            logs.update(val_metrics)
            self.history.update_epoch(logs)
            callbacks.on_epoch_end(self.history.epoch, logs)
            if verbose != 0:
                self.history.display()
        callbacks.on_train_end(self.history._epoch_logs)

    def _do_epoch_(self, dataset, batch_size=1, callbacks=CallbackList(), verbose=0):
        epoch = self.history.epoch
        if verbose == 1:
            num_updates = len(dataset) // batch_size
            if len(dataset) % batch_size > 0:
                num_updates += 1
            pb = tqdm.tqdm(range(num_updates), "Epoch %d" % (epoch))
        elif verbose != 0:
            print("Epoch %d" % (epoch))

        self.train()
        for batch in dataset.to_batch(batch_size):
            self.optimizer.zero_grad()

            callbacks.on_batch_begin(self.history.batch)
            t0 = time.time()
            loss = self._do_batch_(batch)
            t1 = time.time()
            logs = {
                "batch": self.history.batch + 1,
                "loss": loss.cpu().detach().numpy(),
                "time": t1 - t0,
            }
            self.history.update_batch(logs)
            loss.backward()
            callbacks.on_backward_end(self.history.batch)

            self.optimizer.step()
            callbacks.on_batch_end(self.history.batch, logs)
            if verbose == 1:
                pb.set_description(f"Epoch {epoch} loss: {loss:.4f}")
                pb.update()

        if verbose == 1:
            pb.set_description(f"Epoch {epoch}")
            pb.close()
        self.eval()

    def _do_batch_(self, batch):
        loss = torch.tensor(0.0)
        if torch.cuda.is_available():
            loss = loss.cuda()
        num_samples = 0
        for data in batch:
            (x, edge_index), y_true, w = data
            y_true, y_pred, w = self.get_output(data)
            loss += self.loss(y_true, y_pred, w)
            num_samples += 1
        return loss / num_samples

    def evaluate(self, dataset, metrics={}, prefix=None):
        if prefix is not None:
            prefix = prefix + "_"
        else:
            prefix = ""
        metrics["loss"] = self.loss

        logs = {}
        for m in metrics:
            logs[prefix + m] = 0

        self.eval()
        i = 0
        for data in dataset:
            (x, g), y_true, w = data
            y_true, y_pred, w = self.get_output(data)
            for m in metrics:
                val = metrics[m](y_true, y_pred, w).cpu().detach().numpy()
                logs[prefix + m] += val / len(dataset)
        return logs

    def get_output(self, data):
        (x, g), y_true, w = data
        edge_index = to_edge_index(g)
        if torch.cuda.is_available():
            x = x.cuda()
            edge_index = edge_index.cuda()
            y_true = y_true.cuda()
            w = w.cuda()
        x = self.normalize(x, "inputs")
        y_true = self.normalize(y_true, "targets")
        return y_true, self.forward(x, edge_index), w

    def get_weights(self):
        return self.state_dict()

    def setUp(self, dataset):
        self._data_mean = {}
        self._data_var = {}

    def normalize(self, value, key):
        if key in self._data_mean:
            if isinstance(self._data_mean[key], dict):
                assert isinstance(value, dict)
                assert value.keys() == self._data_mean[key].keys()
                for k in value.keys():
                    m = self._data_mean[key][k]
                    v = self._data_var[key][k]
                    if self.using_log:
                        torch.clamp_(value[k], 1e-15)
                        value[k] = torch.log(value[k])
                    value[k] = (value[k] - m) / v ** (0.5)
                return value
            else:
                if self.using_log:
                    torch.clamp_(value, 1e-15)
                    value = torch.log(value)
                return (value - self._data_mean[key]) / self._data_var[key] ** (0.5)
        else:
            if self.using_log:
                torch.clamp_(value, 1e-15)
                value = torch.log(value)
            return value

    def unnormalize(self, value, key):
        if key in self._data_mean:
            if isinstance(self._data_mean[key], dict):
                assert isinstance(value, dict)
                assert value.keys() == self._data_mean.keys()
                for k in value.keys():
                    m = self._data_mean[key][k]
                    v = self._data_var[key][k]
                    value[k] = value[k] * v ** (0.5) + m
                    if self.using_log:
                        value[k] = torch.exp(value[k])
                return value
            else:
                value = value * self._data_var[key] ** (0.5) + self._data_mean[key]
                if self.using_log:
                    value = torch.exp(value)
                return value
        else:
            if self.using_log:
                value = torch.exp(value)
            return value

    def save_weights(self, path):
        state_dict = self.state_dict()
        torch.save(state_dict, path)

    def load_weights(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
        self._data_mean = {}
        self._data_var = {}
        for k in state_dict.keys():
            s = k.split("_")

            if len(s) == 4:
                name, label, key, dkey = s[1], s[2], s[3], None
            elif len(s) == 5:
                name, label, key, dkey = s[1], s[2], s[3], s[4]
            else:
                name, label, key, dkey = None, None, None, None
            if name == "data":
                if label == "mean":
                    if dkey is None:
                        self._data_mean[key] = state_dict[k]
                    else:
                        if dkey in self._data_mean[key]:
                            self._data_mean[key][dkey] = state_dict[k]
                        else:
                            self._data_mean[key] = {dkey: state_dict[k]}
                elif label == "var":
                    if dkey is None:
                        self._data_var[key] = state_dict[k]
                    else:
                        if dkey in self._data_var[key]:
                            self._data_var[key][dkey] = state_dict[k]
                        else:
                            self._data_var[key] = {dkey: state_dict[k]}

    def save_optimizer(self, path):
        state_dict = self.optimizer.state_dict()
        torch.save(state_dict, path)

    def load_optimizer(self, path):
        state_dict = torch.load(path)
        self.optimizer.load_state_dict(state_dict)

    def save_history(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.history, f)

    def load_history(self, path):
        with open(path, "rb") as f:
            self.history = pickle.load(f)

    def num_parameters(self):
        num_params = 0
        for p in self.parameters():
            num_params += torch.tensor(p.size()).prod()
        return num_params
