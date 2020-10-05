import h5py
import json
import networkx as nx
import numpy as np
import os
import pickle
import random
import shutil
import torch
import tqdm
import zipfile

from datetime import datetime
from dynalearn.datasets.getter import get as get_dataset
from dynalearn.dynamics.getter import get as get_dynamics
from dynalearn.experiments.metrics.getter import get as get_metrics
from dynalearn.loggers import LoggerDict, MemoryLogger, TimeLogger, ProgressionLogger
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
        self._dataset = {"main": get_dataset(config.dataset)}
        if "pretrain" in config.__dict__:
            self._dataset["pretrain"] = get_dataset(config.pretrain_dataset)
        self._val_dataset = {}
        self._test_dataset = {}
        self._mode = "main"

        self.networks = get_network(config.networks)
        self.dynamics = get_dynamics(config.dynamics)
        self.model = get_dynamics(config.model)

        # Training related
        self.train_details = config.train_details
        self.metrics = get_metrics(config.metrics)
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
        self.fname_logger = (
            config.fname_logger if "fname_logger" in config.__dict__ else "log.json"
        )

        # Setting seeds
        if "seed" in config.__dict__:
            random.seed(config.seed)
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)

        self.__tasks__ = [
            "load",
            "save",
            "generate_data",
            "partition_val_dataset",
            "partition_test_dataset",
            "train_model",
            "compute_metrics",
            "zip",
        ]
        self.__loggers__ = LoggerDict(
            {
                "time": TimeLogger(),
                "memory": MemoryLogger(),
                "progression": ProgressionLogger(
                    os.path.join(self.path_to_data, "progress")
                ),
            }
        )
        self.__files__ = [
            "config.pickle",
            "loggers.json",
            "data.h5",
            "metrics.h5",
            "history.pickle",
            "model.pt",
            "optim.pt",
        ]

    # Run command
    def run(self, tasks=None, loggers=None):
        tasks = tasks or self.__tasks__
        loggers = loggers or self.__loggers__
        loggers.on_task_begin()
        self.save_config()
        if self.verbose != 0:
            print(f"---Experiment {self.name}---")
            if "time" in loggers.keys():
                begin = loggers["time"].log["begin"]
                print(f"Current time: {begin}")

        for t in tasks:
            if t in self.__tasks__:
                f = getattr(self, t)
                f(loggers=loggers)
            else:
                raise ValueError(
                    f"{t} is an invalid task, possible tasks are {self.__tasks__}"
                )

        loggers.on_task_end()
        self.save(loggers)
        if self.verbose != 0:
            print(f"\n---Finished {self.name}---")
            if "time" in loggers.keys():
                end = loggers["time"].log["end"]
                t = loggers["time"].log["time"]
                print(f"Current time: {end}")
                print(f"Computation time: {t}\n")

    # All tasks
    def train_model(self, loggers=None, save=True, restore_best=True):
        loggers = loggers or LoggerDict()
        if self.verbose != 0:
            print("\n---Training model---")

        self.model.nn.fit(
            self.dataset,
            epochs=self.train_details.epochs,
            batch_size=self.train_details.batch_size,
            val_dataset=self.val_dataset,
            metrics=self.train_metrics,
            callbacks=self.callbacks,
            loggers=loggers,
            verbose=self.verbose,
        )

        if save:
            self.save_model()

        if restore_best:
            self.load_model()

    def generate_data(self, loggers=None, save=True):
        if self.verbose != 0:
            print("\n---Generating data---")
        self.dataset.generate(self)

        if save:
            self.save_data()

    def partition_val_dataset(self, loggers=None, fraction=0.1, bias=0.0):
        self.val_dataset = self.partition_dataset(
            loggers=loggers, fraction=fraction, bias=bias, name="val"
        )

    def partition_test_dataset(self, loggers=None, fraction=0.1, bias=0.0):
        self.test_dataset = self.partition_dataset(
            loggers=loggers, fraction=fraction, bias=bias, name="test"
        )

    def compute_metrics(self, loggers=None, save=True):
        loggers = loggers or LoggerDict()
        if self.verbose != 0:
            print("\n---Computing metrics---")

        if save:
            with h5py.File(join(self.path_to_data, self.fname_metrics), "a") as f:
                for k, m in self.metrics.items():
                    loggers.on_task_midstep("metrics")
                    m.compute(self, verbose=self.verbose)
                    m.save(f)
        else:
            for k, m in self.metrics.items():
                loggers.on_task_midstep("metrics")
                m.compute(self, verbose=self.verbose)

    def zip(self, loggers=None, to_zip=None):
        to_zip = to_zip or self.__files__
        if "config.pickle" not in to_zip:
            to_zip.append("config.pickle")

        zip = zipfile.ZipFile(
            os.path.join(self.path_to_summary, self.name + ".zip"), mode="w"
        )
        for root, _, files in os.walk(self.path_to_data):
            for f in files:
                if f in to_zip:
                    p = os.path.basename(root)
                    zip.write(os.path.join(root, f), os.path.join(p, f))
        zip.close()

    def save(self, loggers=None):
        loggers = loggers or LoggerDict()

        self.save_config()
        self.save_data()
        self.save_model()
        self.save_metrics()
        with open(join(self.path_to_data, self.fname_logger), "w") as f:
            loggers.save(f)

    def load(self, loggers=None):
        self.load_config()
        self.load_data()
        self.load_model()
        self.load_metrics()
        if os.path.exists(join(self.path_to_data, self.fname_logger)):
            with open(join(self.path_to_data, self.fname_logger), "r") as f:
                loggers.load(f)

    # Other methods
    def partition_dataset(self, loggers=None, fraction=0.1, bias=0.0, name="val"):
        loggers = loggers or LoggerDict()
        if self.verbose != 0:
            print(f"\n---Partitioning {name}-data---")

        if f"{name}_fraction" in self.train_details.__dict__:
            fraction = self.train_details.__dict__[f"{name}_fraction"]
        if f"{name}_bias" in self.train_details.__dict__:
            bias = self.train_details.__dict__[f"{name}_bias"]
        partition = self.dataset.partition(fraction, bias=bias)
        if np.sum(partition.network_weights) == 0:
            if self.verbose != 0:
                print("After partitioning, partition is still empty.")
            partition = None
        return partition

    @classmethod
    def from_file(cls, path_to_config):
        with open(path_to_config, "rb") as config_file:
            config = pickle.load(config_file)
        return cls(config)

    @classmethod
    def unzip(cls, path_to_zip, destination=None):
        zip = zipfile.ZipFile(path_to_zip, mode="r")
        path_to_data, _ = os.path.split(zip.namelist()[0])
        destination = destination or "."
        zip.extractall(path=destination)
        cls = cls.from_file(os.path.join(path_to_data, "config.pickle"))
        cls.path_to_data = path_to_data
        cls.load_metrics()
        cls.load_model()
        shutil.rmtree(path_to_data)
        return cls

    def save_data(self):
        with h5py.File(join(self.path_to_data, self.fname_data), "w") as f:
            self.dataset.save(f)

    def save_model(self):
        self.model.nn.save_history(join(self.path_to_data, self.fname_history))
        self.model.nn.save_optimizer(join(self.path_to_data, self.fname_optim))
        self.model.nn.save_weights(join(self.path_to_data, self.fname_model))

    def save_metrics(self):
        with h5py.File(join(self.path_to_data, self.fname_metrics), "a") as f:
            for k, m in self.metrics.items():
                m.save(f)

    def save_config(self):
        with open(join(self.path_to_data, self.fname_config), "wb") as f:
            pickle.dump(self.config, f)

    def load_data(self):
        if exists(join(self.path_to_data, self.fname_data)):
            with h5py.File(join(self.path_to_data, self.fname_data), "r") as f:
                self.dataset.load(f)
        else:
            if self.verbose != 0:
                print("Loading data: Did not find data to load.")

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

    def load_metrics(self):
        if exists(join(self.path_to_data, self.fname_metrics)):
            with h5py.File(join(self.path_to_data, self.fname_metrics), "r") as f:
                for k in self.metrics.keys():
                    self.metrics[k].load(f)
        else:
            if self.verbose != 0:
                print("Loading metrics: Did not find metrics to load.")

    def load_config(self, best=True):
        if exists(join(self.path_to_data, self.fname_config)):
            with open(join(self.path_to_data, self.fname_config), "rb") as f:
                self.config = pickle.load(f)
        else:
            if self.verbose != 0:
                print("Loading config: Did not find config to load.")

    # Other attributes
    @property
    def dataset(self):
        return self._dataset[self._mode]

    @dataset.setter
    def dataset(self, dataset):
        self._dataset[self._mode] = dataset

    @property
    def val_dataset(self):
        if self._mode in self._val_dataset:
            return self._val_dataset[self._mode]
        else:
            return None

    @val_dataset.setter
    def val_dataset(self, val_dataset):
        self._val_dataset[self._mode] = val_dataset

    @property
    def test_dataset(self):
        if self._mode in self._test_dataset:
            return self._test_dataset[self._mode]
        else:
            return None

    @test_dataset.setter
    def test_dataset(self, test_dataset):
        self._test_dataset[self._mode] = test_dataset

    def mode(self, mode):
        if mode in self._dataset:
            self._mode = mode
        else:
            raise ValueError(f"Dataset mode {mode} not available.")
