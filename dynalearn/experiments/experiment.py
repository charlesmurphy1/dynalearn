import h5py
import networkx as nx
import numpy as np
import os
import pickle
import random
import torch

from dynalearn.datasets.getter import get as get_datasets
from dynalearn.dynamics.getter import get as get_dynamics
from dynalearn.experiments.metrics.getter import get as get_post_metrics
from dynalearn.experiments.summaries.getter import get as get_summaries
from dynalearn.networks.getter import get as get_network
from dynalearn.nn.metrics import get as get_train_metrics
from dynalearn.nn.callbacks.getter import get as get_callbacks
from os.path import join, exists


class Experiment:
    def __init__(self, config, verbose=0):
        self.config = config
        self.verbose = verbose
        self.name = config.name

        # Main objects
        self.dataset = get_datasets(config.dataset)
        self.networks = get_network(config.networks)
        self.dynamics = get_dynamics(config.dynamics)
        self.model = get_dynamics(config.model)

        # Training related
        self.val_dataset = None
        self.test_dataset = None
        self.train_details = config.train_details
        self.post_metrics = get_post_metrics(config.post_metrics)
        self.summaries = get_summaries(config.summaries)
        self.train_metrics = get_train_metrics(config.train_metrics)
        self.callbacks = get_callbacks(config.callbacks)

        # Files location
        self.path_to_data = config.path_to_data
        self.path_to_best = config.path_to_best
        self.path_to_summary = config.path_to_summary

        # File names
        self.fname_data = (
            config.fname_data if "fname_data" in config.__dict__ else "data.h5"
        )
        self.fname_model = (
            config.fname_model if "fname_model" in config.__dict__ else "model.pt"
        )
        self.fname_best = (
            config.fname_best
            if "fname_best" in config.__dict__
            else "{self.name}_best.pt"
        )
        self.fname_optim = (
            config.fname_optim if "fname_optim" in config.__dict__ else "optim.pt"
        )
        self.fname_metrics = (
            config.fname_metrics if "fname_metrics" in config.__dict__ else "metrics.h5"
        )
        self.fname_history = (
            config.fname_history
            if "fname_history" in config.__dict__
            else "history.pickle"
        )
        self.fname_config = (
            config.fname_config
            if "fname_config" in config.__dict__
            else "config.pickle"
        )

        # Setting seeds
        if "seed" in config.__dict__:
            random.seed(config.seed)
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)

    @classmethod
    def from_file(cls, path_to_config):
        with open(path_to_config, "rb") as config_file:
            config = pickle.load(config_file)
        return cls(config)

    def run(self):
        if self.verbose != 0:
            print("---Experiment {0}---".format(self.name))
        if self.verbose != 0:
            print("\n---Generating data---")
        self.generate_data()

        if self.verbose != 0:
            print("\n---Training model---")
        self.train_model()

        if self.verbose != 0:
            print("\n---Computing metrics---")
        self.compute_metrics()

        if self.verbose != 0:
            print("\n---Summarizing---")
        self.compute_summaries()

        if self.verbose != 0:
            print("\n---Saving---")
        self.save()
        return

    def generate_data(self):
        self.dataset.generate(self)
        if "val_fraction" in self.train_details.__dict__:
            if self.verbose != 0:
                print("Partitioning for validation set")
            p = self.train_details.val_fraction
            b = self.train_details.val_bias
            self.val_dataset = self.dataset.partition(p, bias=b)
            if np.sum(self.val_dataset.network_weights) == 0:
                if self.verbose != 0:
                    print("After partitioning, validation set is still empty.")
                self.val_dataset = None
        if "test_fraction" in self.train_details.__dict__:
            if self.verbose != 0:
                print("Partitioning for test set")
            p = self.train_details.test_fraction
            b = self.train_details.test_bias
            self.test_dataset = self.dataset.partition(p, bias=b)
            if np.sum(self.val_dataset.network_weights) == 0:
                if self.verbose != 0:
                    print("After partitioning, test set is still empty.")
                self.val_dataset = None

    def train_model(self):
        self.model.nn.fit(
            self.dataset,
            epochs=self.train_details.epochs,
            batch_size=self.train_details.batch_size,
            val_dataset=self.val_dataset,
            metrics=self.train_metrics,
            callbacks=self.callbacks,
            verbose=self.verbose,
        )

    def compute_metrics(self):
        for k, m in self.post_metrics.items():
            m.compute(self, verbose=self.verbose)

    def compute_summaries(self):
        for k, m in self.summaries.items():
            m.compute(self, verbose=self.verbose)

    def save(self):
        with h5py.File(join(self.path_to_data, self.fname_data), "w") as f:
            self.dataset.save(f)
        with h5py.File(join(self.path_to_data, self.fname_metrics), "w") as f:
            for k, m in self.post_metrics.items():
                m.save(f)
        with h5py.File(join(self.path_to_summary, self.name + ".h5"), "w") as f:
            for k, m in self.summaries.items():
                m.save(f)
        with open(join(self.path_to_data, self.fname_config), "wb") as f:
            pickle.dump(self.config, f)

        self.model.nn.save_history(join(self.path_to_data, self.fname_history))
        self.model.nn.save_optimizer(join(self.path_to_data, self.fname_optim))
        self.model.nn.save_weights(join(self.path_to_data, self.fname_model))

    def load(self, best=True):
        if exists(join(self.path_to_data, self.fname_data)):
            with h5py.File(join(self.path_to_data, self.fname_data), "r") as f:
                self.dataset.load(f)
        if exists(join(self.path_to_data, self.fname_metrics)):
            with h5py.File(join(self.path_to_data, self.fname_metrics), "r") as f:
                for k in self.post_metrics.keys():
                    self.post_metrics[k].load(f)
        if exists(join(self.path_to_summary, self.name + ".h5")):
            with h5py.File(join(self.path_to_summary, self.name + ".h5"), "r") as f:
                for k in self.summaries.keys():
                    self.summaries[k].load(f)
        if exists(join(self.path_to_data, self.fname_config)):
            with open(join(self.path_to_data, self.fname_config), "rb") as f:
                self.config = pickle.load(f)

        if exists(join(self.path_to_data, self.fname_optim)):
            self.model.nn.load_history(join(self.path_to_data, self.fname_optim))
        if exists(join(self.path_to_data, self.fname_optim)):
            self.model.nn.load_optimizer(join(self.path_to_data, self.fname_optim))

        if best and exists(join(self.path_to_best, self.fname_best)):
            self.model.nn.load_weights(join(self.path_to_best, self.fname_best))
        elif join(self.path_to_best, self.fname_model):
            self.model.nn.load_weights(join(self.path_to_data, self.fname_model))
