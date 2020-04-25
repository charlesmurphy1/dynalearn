import networkx as nx
import numpy as np
import pickle
import time
import torch
import tqdm

from dynalearn.nn.callbacks import CallbackList
from dynalearn.nn.history import History
from dynalearn.nn.loss import get as get_loss
from dynalearn.nn.optimizer import get as get_optimizer
from dynalearn.utilities import to_edge_index, Config
from abc import abstractmethod


class GraphNeuralNetwork(torch.nn.Module):
    def __init__(self, config=None, **kwargs):
        torch.nn.Module.__init__(self)
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        self.loss = get_loss(config.loss)
        self.optimizer = get_optimizer(config.optimizer)
        self.history = History()
        if torch.cuda.is_available():
            self = self.cuda()

    @abstractmethod
    def forward(self, x, edge_index):
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
        for i in range(epochs):
            callbacks.on_epoch_begin(self.history.epoch)
            t0 = time.time()
            self._do_epoch_(
                dataset, batch_size=batch_size, callbacks=callbacks, verbose=verbose,
            )
            t1 = time.time()

            logs = {"epoch": self.history.epoch + 1, "time": t1 - t0}
            logs.update(self.evaluate(dataset, metrics))
            if val_dataset is not None:
                logs.update(self.evaluate(val_dataset, metrics, "val"))
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
        num_samples = 0
        for data in batch:
            (x, edge_index), y_true, w = data
            if torch.cuda.is_available():
                x = x.cuda()
                edge_index = edge_index.cuda()
                y_true = y_true.cuda()
                w = w.cuda()
            y_pred = self.forward(x, edge_index)
            loss += self.loss(y_pred, y_true, w)
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
        for data in dataset:
            (x, edge_index), y_true, w = data
            if torch.cuda.is_available():
                x = x.cuda()
                edge_index = edge_index.cuda()
                y_true = y_true.cuda()
                w = w.cuda()
            y_pred = self.forward(x, edge_index)
            for m in metrics:
                val = metrics[m](y_pred, y_true, w).cpu().detach().numpy()
                logs[prefix + m] += val / len(dataset)
        return logs

    def get_weights(self):
        return self.state_dict()

    def save_weights(self, path):
        state_dict = self.state_dict()
        torch.save(state_dict, path)

    def load_weights(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)

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
