import copy
import dynalearn as dl
import h5py
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy


class Experiment:
    def __init__(self, config, verbose=1):
        self.__config = config
        self.graph_model = dl.graphs.get(config["graph"])
        self.dynamics_model = dl.dynamics.get(config["dynamics"])
        self.model = dl.models.get(config["model"])
        self.model.num_nodes = self.graph_model.num_nodes
        self.generator = dl.datasets.get(
            config["generator"], self.graph_model, self.dynamics_model
        )
        self.metrics = dl.metrics.get(config["metrics"], self.dynamics_model)

        self.name = config["name"]
        self.path_to_dir = config["path_to_dir"]
        self.path_to_bestmodel = config["path_to_bestmodel"]
        if "filename_data" not in config:
            self.filename_data = "data.h5"
        else:
            self.filename_data = config["filename_data"]

        if "filename_metrics" not in config:
            self.filename_metrics = "data.h5"
        else:
            self.filename_metrics = config["filename_metrics"]

        if "filename_model" not in config:
            self.filename_model = "data.h5"
        else:
            self.filename_model = config["filename_model"]

        if "filename_history" not in config:
            self.filename_history = "data.h5"
        else:
            self.filename_history = config["filename_history"]

        self.num_graphs = config["training"].num_graphs
        self.num_samples = config["training"].num_samples
        self.epoch = 0
        self.num_epochs = config["training"].num_epochs
        self.optimizer = config["training"].optimizer

        self.callbacks = config["training"].callbacks
        self.callbacks.append(
            ks.callbacks.ModelCheckpoint(
                os.path.join(self.path_to_bestmodel, self.name + ".h5"),
                save_best_only=True,
                monitor="val_loss",
                mode="min",
                period=1,
                verbose=1,
            )
        )
        self.training_metrics = config["training"].training_metrics

        if not os.path.exists(os.path.join(self.path_to_dir, self.name)):
            os.makedirs(os.path.join(self.path_to_dir, self.name))

        if not os.path.exists(os.path.join(self.path_to_bestmodel)):
            os.makedirs(os.path.join(self.path_to_bestmodel))

        self.verbose = verbose
        self.history = dict()

        return

    def run(self):
        if self.verbose:
            print("\n---Generating data---")
        self.generate_data()

        if self.verbose:
            print("\n---Training model---")
        self.train_model()
        self.load_model(best=True)

        if self.verbose:
            print("\n---Computing metrics---")
        self.compute_metrics()

        if self.verbose:
            print("\n---Saving all---")
        self.save(True)

    def save(self, overwrite=False):
        self.save_data(overwrite)
        self.save_model(overwrite)
        self.save_metrics(overwrite)
        self.save_history(overwrite)

    def load(self):
        self.load_data()
        self.load_model()
        self.load_metrics()
        self.load_history()

    def generate_data(self):
        for i in range(self.__config["training"].num_graphs):
            self.generator.generate(self.__config["training"].num_samples)

        if self.__config["training"].val_fraction is not None:
            val_fraction = self.__config["training"].val_fraction
            val_bias = self.__config["training"].val_bias
            if self.verbose:
                print("Partitioning generator for validation")
            self.generator.partition_sampler("val", val_fraction, val_bias)

        if self.__config["training"].test_fraction is not None:
            test_fraction = self.__config["training"].test_fraction
            test_bias = self.__config["training"].test_bias
            if self.verbose:
                print("Partitioning generator for test")
            self.generator.partition_sampler("test", test_fraction, test_bias)

    def train_model(self, epochs=None):

        if epochs is None:
            epochs = self.num_epochs

        i_epoch = self.epoch
        f_epoch = self.epoch + epochs

        self.model.model.compile(
            self.optimizer, self.model.loss_fct, self.training_metrics
        )

        if "val" in self.generator.samplers:
            val_generator = copy.deepcopy(self.generator)
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

    def save_model(self, overwrite=False):
        self.model.model.save_weights(
            os.path.join(self.path_to_dir, self.name, self.filename_model)
        )

    def load_model(self, best=True):
        if best:
            path = os.path.join(self.path_to_bestmodel, self.name + ".h5")
        else:
            path = os.path.join(self.path_to_dir, self.name, self.filename_model)
        if os.path.exists(path):
            self.model.model.load_weights(path)

    def save_data(self, overwrite=False):
        h5file = h5py.File(
            os.path.join(self.path_to_dir, self.name, self.filename_data)
        )
        self.generator.save(h5file, overwrite)
        h5file.close()

    def load_data(self):
        path = os.path.join(self.path_to_dir, self.name, self.filename_data)
        if os.path.exists(path):
            h5file = h5py.File(path)
        else:
            return

        self.generator.load(h5file)
        h5file.close()

    def save_history(self, overwrite=False):
        path = os.path.join(self.path_to_dir, self.name, self.filename_history)
        if os.path.exists(path) and not overwrite:
            return
        h5file = h5py.File(path, "w")
        for k, v in self.history.items():
            h5file.create_dataset(k, data=v, fillvalue=np.nan)

    def load_history(self,):
        path = os.path.join(self.path_to_dir, self.name, self.filename_history)
        if os.path.exists(path):
            h5file = h5py.File(path)
        else:
            return

        for k, v in h5file.items():
            self.history[k] = v[...]
        h5file.close()

    def save_metrics(self, overwrite=False):
        path = os.path.join(self.path_to_dir, self.name, self.filename_metrics)
        if os.path.exists(path) and not overwrite:
            return

        h5file = h5py.File(path, "w")
        for k, v in self.metrics.items():
            v.save(k, h5file)
        h5file.close()

    def load_metrics(self):
        path = os.path.join(self.path_to_dir, self.name, self.filename_metrics)
        if os.path.exists(path):
            h5file = h5py.File(path)
        else:
            return

        for k, v in self.metrics.items():
            v.load(k, h5file["metrics"])

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

    @property
    def path_to_dir(self):
        return self._path_to_dir

    @path_to_dir.setter
    def path_to_dir(self, path_to_dir):
        self._path_to_dir = path_to_dir
        if not os.path.exists(path_to_dir):
            os.makedirs(path_to_dir)
