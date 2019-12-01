import copy
import dynalearn as dl
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras as ks


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy


class Experiment:
    def __init__(self, param_dict):
        self.param_dict = param_dict
        self.graph_model = dl.graphs.get(param_dict["graph"])
        self.dynamics_model = dl.dynamics.get(param_dict["dynamics"])
        self.model = dl.models.get(
            param_dict["model"], self.graph_model, self.dynamics_model
        )
        self.generator = dl.generators.get(
            param_dict["generator"], self.graph_model, self.dynamics_model
        )
        self.metrics = dl.metrics.get(param_dict["metrics"], self.dynamics_model)

        self.configure(param_dict["config"])

        self.path_to_data = param_dict["path_to_data"]
        self.path_to_metrics = param_dict["path_to_metrics"]
        self.path_to_model = param_dict["path_to_model"]
        self.path_to_best = param_dict["path_to_best"]

        return

    def configure(self, params):
        np.random.seed(params["np_seed"])

        self.optimizer = ks.optimizers.get(params["optimizer"])
        self.optimizer.lr = K.variable(params["initial_lr"])

        if params["loss"] == "noisy_crossentropy":
            self.loss = get_noisy_crossentropy(noise=params["target_noise"])
        else:
            self.loss = ks.losses.get(params["loss"])

        self.callbacks = [
            ks.callbacks.LearningRateScheduler(
                dl.utilities.get_schedule(params["schedule"]), verbose=1
            ),
            ks.callbacks.ModelCheckpoint(
                os.path.join(self.path_to_best),
                save_best_only=True,
                monitor="val_loss",
                mode="min",
                period=1,
                verbose=1,
            ),
        ]

        self.training_metrics = [dl.utilities.metrics.model_entropy]

    def generate_data(self):
        num_graphs = self.param_dict["generator"]["params"]["num_graphs"]
        num_sample = self.param_dict["generator"]["params"]["num_sample"]
        resampling_time = self.param_dict["generator"]["params"]["resampling_time"]

        for i in range(num_graphs):
            self.generator.generate(num_sample, resampling_time)

        if "val_fraction" in self.param_dict["generate"]["params"]:
            val_fraction = self.param_dict["generator"]["params"]["val_fraction"]
            val_bias = self.param_dict["generator"]["params"]["val_bias"]
            if self.verbose:
                print("Partitioning generator for validation")
            self.generator.partition_sampler("val", val_fraction, val_bias)

        if "test_fraction" in self.param_dict["generate"]["params"]:
            test_fraction = self.param_dict["generator"]["params"]["test_fraction"]
            test_bias = self.param_dict["generator"]["params"]["test_bias"]
            if self.verbose:
                print("Partitioning generator for test")
            self.generator.partition_sampler("test", test_fraction, test_bias)

    def train_model(self, epochs=None):

        if epochs is None:
            epochs = self.epochs

        i_epoch = self.epoch
        f_epoch = self.epoch + epochs

        self.model.model.compile(self.optimizer, self.loss, self.training_metrics)

        if "val" in self.generator.samplers:
            val_generator = copy.deepcopy(self.generator)
            val_generator.mode = "val"
        else:
            val_generator = None

        history = self.model.model.fit_generator(
            self.generator,
            validation_data=val_generator,
            steps_per_epoch=self.param_dict["generator"]["params"]["num_sample"],
            validation_steps=val_steps,
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

    def save_model(self):
        self.model.model.save_weights(self.path_to_model)

    def load_model(self):
        self.model.model.load_weights(self.path_to_model)

    def save_metrics(self, overwrite=False):
        h5file = h5py.File(self.path_to_metrics)
        _save_metrics = True
        _save_history = True

        if "metrics" in h5file:
            if not overwrite:
                _save_metrics = False
            else:
                del h5file["metrics"]

        if "history" in h5file:
            if not overwrite:
                _save_history = False
            else:
                del h5file["history"]

        if _save_metrics:
            h5file.create_group("metrics")
            h5group = h5file["metrics"]
            for k, v in self.metrics.items():
                v.save(k, h5group)

        if _save_history:
            h5group = h5file["history"]
            for k, v in self.history.items():
                h5group.create_dataset(k, data=v, fillvalue=np.nan)

    def load_metrics(self):
        h5file = h5py.File(self.path_to_metrics)
        _load_metrics = True
        _load_history = True

        if "metrics" not in h5file:
            _load_metrics = False
        if "history" not in h5file:
            _load_history = False

        for k, v in self.metrics.items():
            v.load(k, h5file["metrics"])

        for k, v in h5file["history"].items():
            self.history[k] = list(v[...])

    def save_data(self, overwrite=False):
        h5file = h5py.File(self.path_to_data)
        self.generator.save(h5file, overwrite)
        h5file.close()

    def load_data(self):
        h5file = h5py.File(self.path_to_data)
        self.generator.load(h5file)
