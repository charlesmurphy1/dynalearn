import h5py
import networkx as nx
import numpy as np
import os
import pickle
import random
import torch
import tqdm

from datetime import datetime
from dynalearn.datasets.getter import get as get_datasets
from dynalearn.dynamics.getter import get as get_dynamics
from dynalearn.experiments.metrics.getter import get as get_metrics
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
        self.metrics = get_metrics(config.metrics)
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
            begin = datetime.now()
            print(f"---Experiment {self.name}---")
            print(f"Current time: {begin.strftime('%Y-%m-%d %H:%M:%S')}")

        if self.verbose != 0:
            print("\n---Generating data---")
        self.generate_data()
        self.save_data()

        if self.verbose != 0:
            print("\n---Training model---")
        self.train_model()
        self.save_model()

        if self.verbose != 0:
            print("\n---Computing metrics---")
        self.load_model(restore_best=True)
        self.compute_metrics()
        self.save_metrics()

        if self.verbose != 0:
            print("\n---Summarizing---")
        self.compute_summaries()
        self.save_summaries()

        if self.verbose != 0:
            end = datetime.now()
            print(f"\n---Finished {self.name}---")
            print(f"Current time: {end.strftime('%Y-%m-%d %H:%M:%S')}")
            dt = end - begin
            days = dt.seconds // (24 * 60 * 60)
            hours = dt.seconds // (60 * 60) - (days * 24)
            mins = dt.seconds // 60 - (days * 24 + hours) * 60
            secs = dt.seconds - ((days * 24 + hours) * 60 + mins) * 60
            print(f"Computation time: {days:0=2d}-{hours:0=2d}:{mins:0=2d}:{secs:0=2d}")

    def save(self):
        self.save_data()
        self.save_model()
        self.save_metrics()
        self.save_summaries()
        self.save_config()

    def load(self):
        self.load_data()
        self.load_model()
        self.load_metrics()
        self.load_summaries()
        self.load_config()

    def generate_data(self):
        self.dataset.generate(self)

        if "val_fraction" in self.train_details.__dict__:
            if self.verbose != 0 and self.verbose != 1:
                print("Partitioning for validation set")
            elif self.verbose == 1:
                pb = tqdm.tqdm(
                    range(len(self.dataset)), "Partitioning for validation set"
                )
            else:
                pb = None
            p = self.train_details.val_fraction
            b = self.train_details.val_bias
            self.val_dataset = self.dataset.partition(p, bias=b, pb=pb)
            if np.sum(self.val_dataset.network_weights) == 0:
                if self.verbose != 0:
                    print("After partitioning, validation set is still empty.")
                self.val_dataset = None

        if "test_fraction" in self.train_details.__dict__:
            if self.verbose != 0 and self.verbose != 1:
                print("Partitioning for test set")
            elif self.verbose == 1:
                pb = tqdm.tqdm(range(len(self.dataset)), "Partitioning for test set")
            else:
                pb = None
            p = self.train_details.test_fraction
            b = self.train_details.test_bias
            self.test_dataset = self.dataset.partition(p, bias=b, pb=pb)
            if np.sum(self.val_dataset.network_weights) == 0:
                if self.verbose != 0:
                    print("After partitioning, test set is still empty.")
                self.val_dataset = None

    def save_data(self):
        with h5py.File(join(self.path_to_data, self.fname_data), "w") as f:
            self.dataset.save(f)

    def load_data(self):
        if exists(join(self.path_to_data, self.fname_data)):
            with h5py.File(join(self.path_to_data, self.fname_data), "r") as f:
                self.dataset.load(f)
        else:
            if self.verbose != 0:
                print("Loading data: Did not find data to load.")

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

    def save_model(self):
        self.model.nn.save_history(join(self.path_to_data, self.fname_history))
        self.model.nn.save_optimizer(join(self.path_to_data, self.fname_optim))
        self.model.nn.save_weights(join(self.path_to_data, self.fname_model))

    def load_model(self, restore_best=True):
        if exists(join(self.path_to_data, self.fname_history)):
            self.model.nn.load_history(join(self.path_to_data, self.fname_history))
        else:
            if self.verbose != 0:
                print("Loading model: Did not find history to load.")

        if exists(join(self.path_to_data, self.fname_optim)):
            self.model.nn.load_optimizer(join(self.path_to_data, self.fname_optim))
        else:
            if self.verbose != 0:
                print("Loading model: Did not find optimizer to load.")

        if restore_best and exists(self.path_to_best):
            self.model.nn.load_weights(self.path_to_best)
        elif exists(join(self.path_to_data, self.fname_model)):
            self.model.nn.load_weights(join(self.path_to_data, self.fname_model))
        else:
            if self.verbose != 0:
                print("Loading model: Did not find model to load.")

    def compute_metrics(self):
        for k, m in self.metrics.items():
            m.compute(self, verbose=self.verbose)

    def save_metrics(self):
        with h5py.File(join(self.path_to_data, self.fname_metrics), "a") as f:
            for k, m in self.metrics.items():
                m.save(f)

    def load_metrics(self):
        if exists(join(self.path_to_data, self.fname_metrics)):
            with h5py.File(join(self.path_to_data, self.fname_metrics), "r") as f:
                for k in self.metrics.keys():
                    self.metrics[k].load(f)
        else:
            if self.verbose != 0:
                print("Loading metrics: Did not find metrics to load.")

    def compute_summaries(self):
        for k, m in self.summaries.items():
            m.compute(self, verbose=self.verbose)

    def save_summaries(self):
        with h5py.File(join(self.path_to_summary, self.name + ".h5"), "a") as f:
            for k, m in self.summaries.items():
                m.save(f)

    def load_summaries(self):
        if exists(join(self.path_to_summary, self.name + ".h5")):
            with h5py.File(join(self.path_to_summary, self.name + ".h5"), "r") as f:
                for k in self.summaries.keys():
                    self.summaries[k].load(f)
        else:
            if self.verbose != 0:
                print("Loading summaries: Did not find summaries to load.")

    def save_config(self):
        with open(join(self.path_to_data, self.fname_config), "wb") as f:
            pickle.dump(self.config, f)

    def load_config(self, best=True):
        if exists(join(self.path_to_data, self.fname_config)):
            with open(join(self.path_to_data, self.fname_config), "rb") as f:
                self.config = pickle.load(f)
        else:
            if self.verbose != 0:
                print("Loading config: Did not find config to load.")
