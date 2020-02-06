import dynalearn as dl
import h5py
import numpy as np
import os
import pickle
import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K

from tensorflow.keras.optimizers import get
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.backend import variable


class Experiment:
    def __init__(self, config, verbose=0):
        self.__config = config
        self._verbose = verbose
        self.graph_model = dl.graphs.get(config["graph"])
        self.dynamics_model = dl.dynamics.get(config["dynamics"])
        self.model = dl.models.get(config["model"])
        self.model.num_nodes = self.graph_model.num_nodes
        self.generator = dl.datasets.get(
            config["generator"], self.graph_model, self.dynamics_model
        )
        self.metrics = dl.metrics.get(config["metrics"])

        self.name = config["name"]
        self.path_to_data = config["path_to_data"]
        self.path_to_models = config["path_to_models"]

        if "filename_config" not in config:
            self.filename_config = "config.pickle"
        else:
            self.filename_data = config["filename_data"]
        if "filename_data" not in config:
            self.filename_data = "data.h5"
        else:
            self.filename_data = config["filename_data"]

        if "filename_metrics" not in config:
            self.filename_metrics = "metrics.h5"
        else:
            self.filename_metrics = config["filename_metrics"]

        if "filename_model" not in config:
            self.filename_model = "model.h5"
        else:
            self.filename_model = config["filename_model"]

        if "filename_history" not in config:
            self.filename_history = "history.h5"
        else:
            self.filename_history = config["filename_history"]

        self.num_graphs = config["training"].num_graphs
        self.num_samples = config["training"].num_samples
        self.epoch = 0
        self.num_epochs = config["training"].num_epochs

        self.optimizer = ks.optimizers.get(config["training"].name_optimizer)
        self.optimizer.lr = variable(config["training"].initial_lr)
        self.callbacks = [
            LearningRateScheduler(
                dl.utilities.get_schedule(config["training"].schedule), verbose=0,
            ),
            ks.callbacks.ModelCheckpoint(
                os.path.join(self.path_to_models, self.name + ".h5"),
                save_best_only=True,
                monitor="val_loss",
                mode="min",
                period=1,
                verbose=0,
            ),
        ]

        self.training_metrics = [
            dl.utilities.get_metrics(m) for m in config["training"].training_metrics
        ]

        if not os.path.exists(self.path_to_data):
            os.makedirs(self.path_to_data)

        if not os.path.exists(os.path.join(self.path_to_models)):
            os.makedirs(os.path.join(self.path_to_models))

        self.verbose = verbose
        self.history = dict()

        np.random.seed(config["training"].np_seed)

        return

    @classmethod
    def from_file(cls, path_to_config):
        with open(path_to_config, "rb") as config_file:
            config = pickle.load(config_file)
        return cls(config)

    def run(self, overwrite=True):
        self.save_config()
        if self.verbose != 0:
            print("\n---Building---")
        if self.verbose != 0:
            print("\n---Generating data---")
        self.generate_data()
        self.save_data(overwrite)

        if self.verbose != 0:
            print("\n---Training model---")
        self.train_model()
        self.save_history(overwrite)
        self.save_model(overwrite)
        self.load_model(best=True)

        if self.verbose != 0:
            print("\n---Computing metrics---")
        self.compute_metrics()
        self.save_metrics(overwrite)

        if self.verbose != 0:
            print("\n---Finished---")

    def save(self, overwrite=True):
        self.save_config(overwrite)
        self.save_data(overwrite)
        self.save_model(overwrite)
        self.save_metrics(overwrite)
        self.save_history(overwrite)

    def load(self):
        # self.load_config()
        self.load_data()
        self.load_model()
        self.load_metrics()
        self.load_history()

    def save_config(self, overwrite=True):
        path = os.path.join(self.path_to_data, self.filename_config)
        if os.path.exists(path) and not overwrite:
            return

        with open(path, "wb") as f:
            pickle.dump(self.__config, f)

    def load_config(self,):
        path = os.path.join(self.path_to_data, self.filename_config)

        if not os.path.exists(path):
            return

        with open(path, "rb") as f:
            self.__config = pickle.load(f)

    def generate_data(self):
        for i in range(self.__config["training"].num_graphs):
            self.generator.generate(self.__config["training"].num_samples)
        if self.__config["training"].val_fraction is not None:
            val_fraction = self.__config["training"].val_fraction
            val_bias = self.__config["training"].val_bias
            if self.verbose != 0:
                print("Partitioning generator for validation")
            self.generator.partition_sampler(
                "val", fraction=val_fraction, bias=val_bias
            )

        if self.__config["training"].test_fraction is not None:
            test_fraction = self.__config["training"].test_fraction
            test_bias = self.__config["training"].test_bias
            if self.verbose != 0:
                print("Partitioning generator for test")
            self.generator.partition_sampler(
                "test", fraction=test_fraction, bias=test_bias
            )

    def train_model(self, epochs=None):

        if epochs is None:
            epochs = self.num_epochs

        i_epoch = self.epoch
        f_epoch = self.epoch + epochs

        self.model.model.compile(
            self.optimizer, self.model.loss_fct, self.training_metrics
        )

        if "val" in self.generator.samplers:
            val_generator = self.generator.copy()
            val_generator.mode = "val"
        else:
            val_generator = None
        history = self.model.model.fit_generator(
            self.generator,
            validation_data=val_generator,
            steps_per_epoch=self.__config["training"].step_per_epoch,
            validation_steps=self.__config["training"].step_per_epoch,
            initial_epoch=i_epoch,
            epochs=f_epoch,
            verbose=self.verbose,
            callbacks=self.callbacks,
        )

        for k in history.history:
            if k not in self.history:
                self.history[k] = [np.nan] * i_epoch
            elif len(self.history[k]) < i_epoch:
                to_fill = i_epoch - len(self.history[k])
                self.history[k].extend([np.nan] * to_fill)
            self.history[k].extend(history.history[k])

        self.epoch += epochs

    def compute_metrics(self):
        for k, m in self.metrics.items():
            m.compute(self)

    def save_model(self, overwrite=True):
        self.model.model.save_weights(
            os.path.join(self.path_to_data, self.filename_model)
        )

    def load_model(self, best=True):
        if best:
            path = os.path.join(self.path_to_models, self.name + ".h5")
        else:
            path = os.path.join(self.path_to_data, self.filename_model)
        if os.path.exists(path):
            self.model.model.load_weights(path)

    def save_data(self, overwrite=True):
        path = os.path.join(self.path_to_data, self.filename_data)
        if os.path.exists(path) and not overwrite:
            return
        h5file = h5py.File(path, "w")
        self.generator.save(h5file, overwrite)
        h5file.close()

    def load_data(self):
        path = os.path.join(self.path_to_data, self.filename_data)
        if os.path.exists(path):
            h5file = h5py.File(path, "r")
        else:
            return

        self.generator.load(h5file)
        h5file.close()

    def save_history(self, overwrite=True):
        path = os.path.join(self.path_to_data, self.filename_history)
        if os.path.exists(path) and not overwrite:
            return
        h5file = h5py.File(path, "w")
        for k, v in self.history.items():
            h5file.create_dataset(k, data=v, fillvalue=np.nan)

    def load_history(self,):
        path = os.path.join(self.path_to_data, self.filename_history)
        if os.path.exists(path):
            h5file = h5py.File(path, "r")
        else:
            return

        for k, v in h5file.items():
            self.history[k] = v[...]
        h5file.close()

    def save_metrics(self, overwrite=True):
        path = os.path.join(self.path_to_data, self.filename_metrics)
        if os.path.exists(path) and not overwrite:
            return

        h5file = h5py.File(path, "a")
        for k, v in self.metrics.items():
            v.save(k, h5file, overwrite=overwrite)
        h5file.close()

    def load_metrics(self):
        path = os.path.join(self.path_to_data, self.filename_metrics)
        if os.path.exists(path):
            h5file = h5py.File(path, "r")
        else:
            return

        for k, v in self.metrics.items():
            v.load(k, h5file)

        h5file.close()

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, verbose):
        self._verbose = verbose
        self.generator.verbose = verbose
        for m in self.metrics:
            self.metrics[m].verbose = verbose
