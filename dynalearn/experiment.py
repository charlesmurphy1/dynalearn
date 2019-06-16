import copy
import dynalearn as dl
import h5py
import numpy as np
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
import tensorflow.keras.backend as K


class Experiment:
    def __init__(
        self,
        name,
        model,
        generator,
        loss=categorical_crossentropy,
        optimizer=Adam,
        metrics=["accuracy"],
        learning_rate=1e-4,
        callbacks=None,
        numpy_seed=1,
        tensorflow_seed=2,
        verbose=1,
    ):

        self.name = name
        self.history = {}
        self.epoch = 0

        self.model = model
        self.generator = generator
        self.val_generator = None
        self.test_generator = None
        self.graph_model = self.generator.graph_model
        self.dynamics_model = self.generator.dynamics_model

        self.loss = loss
        self.optimizer = optimizer
        self.optimizer.lr = K.variable(learning_rate)
        self.metrics = metrics
        self.callbacks = callbacks
        self.np_seed = numpy_seed
        self.tf_seed = tensorflow_seed
        self.verbose = verbose

        self.model.model.compile(self.optimizer, self.loss, self.metrics)

    def generate_data(
        self, num_graphs, num_sample, T, val_fraction=None, test_fraction=None, **kwargs
    ):
        for i in range(num_graphs):
            self.generator.generate(num_sample, T, **kwargs)

        if val_fraction is not None:
            self.val_generator = self.generator.parition_generator(val_fraction)
            self.val_generator.sampler = dl.generators.SequentialSampler()

        if test_fraction is not None:
            self.test_generator = self.generator.parition_generator(test_fraction)
            self.test_generator.sampler = dl.generators.SequentialSampler()

    def train_model(self, epochs, steps_per_epoch, validation_steps=0):
        if self.val_generator is None:
            validation_steps = None
        i_epoch = self.epoch
        f_epoch = self.epoch + epochs
        history = self.model.model.fit_generator(
            self.generator,
            validation_data=self.val_generator,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            initial_epoch=i_epoch,
            epochs=f_epoch,
            verbose=self.verbose,
            callbacks=self.callbacks,
        )

        for k in history.history:
            if k not in self.history:
                self.history[k] = [None] * i_epoch
            elif len(self.history[k]) < i_epoch:
                to_fill = i_epoch - len(self.history[k])
                self.history[k].extend([None] * to_fill)
            self.history[k].extend(history.history[k])

        self.epoch += epochs

        return history

    def save_hdf5_model(self, h5file):
        model_name = type(self.model).__name__
        model_params = self.model.params
        for name, value in model_params.items():
            h5file.create_dataset("/model/params/" + name, data=value)

        h5file.create_dataset("/model/name", data=model_name)
        weights = self.model.model.get_weights()

        for i, w in enumerate(weights):
            h5file.create_dataset("/model/weights/w_" + str(i), data=w)

    def load_hdf5_model(self, h5file):
        if str(h5file["/model/name/"][...]) != type(self.model).__name__:
            raise ValueError("invalid type for model during loading.")

        weights = [None for i in h5file["/model/weights"].keys()]
        for k, v in h5file["/model/weights"].items():
            weights[int(k[2:])] = np.array(v)

        self.model.model.set_weights(weights)
        return 0

    def load_best_weights(self, path):
        self.model.model.load_weights(path)
        return 0

    def save_hdf5_optimizer(self, h5file):
        optimizer_name = type(self.optimizer).__name__
        optimizer_params = self.optimizer.get_config()
        h5file.create_dataset("/optimizer/name", data=optimizer_name)

        for name, value in optimizer_params.items():
            h5file.create_dataset("/optimizer/params/" + name, data=value)

        h5file.create_dataset("/loss", data=self.loss.__name__)

    def save_hdf5_history(self, h5file):
        for name, value in self.history.items():
            h5file.create_dataset("/history/" + name, data=value, fillvalue=np.nan)

    def load_hdf5_history(self, h5file):
        for k, v in h5file["/history/"].items():
            self.history[k] = list(v[...])

    def save_hdf5_data(self, h5file):
        graph_name = type(self.graph_model).__name__
        graph_params = self.graph_model.params

        dynamics_name = type(self.dynamics_model).__name__
        dynamics_params = self.dynamics_model.params

        h5file.create_dataset("/graph/name", data=graph_name)
        for name, value in graph_params.items():
            h5file.create_dataset("/graph/params/" + name, data=value)

        h5file.create_dataset("/dynamics/name", data=dynamics_name)
        for name, value in dynamics_params.items():
            h5file.create_dataset("/dynamics/params/" + name, data=value)

        for g_name in self.generator.graphs:
            inputs = self.generator.inputs[g_name]
            adj = self.generator.graphs[g_name]
            targets = self.generator.targets[g_name]
            h5file.create_dataset("/data/" + g_name + "/adj_matrix", data=adj)
            h5file.create_dataset("/data/" + g_name + "/inputs", data=inputs)
            h5file.create_dataset("/data/" + g_name + "/targets", data=targets)
            train_node_set = self.generator.sampler.avail_node_set[g_name]
            h5file.create_dataset(
                "/data/" + g_name + "/train_node_set", data=train_node_set
            )
            if self.generator.sampler.with_weights:
                node_weights = self.generator.sampler.node_weights[g_name]
                h5file.create_dataset(
                    "/data/" + g_name + "/node_weights", data=node_weights
                )
                state_weights = np.array(
                    [
                        self.generator.sampler.state_weights[g_name][i]
                        for i in self.generator.sampler.avail_state_set[g_name]
                    ]
                )
                h5file.create_dataset(
                    "/data/" + g_name + "/state_weights", data=state_weights
                )
                graph_weights = self.generator.sampler.graph_weights[g_name]
                h5file.create_dataset(
                    "/data/" + g_name + "/graph_weights", data=graph_weights
                )
            if self.val_generator is not None:
                val_node_set = self.val_generator.sampler.avail_node_set[g_name]
                h5file.create_dataset(
                    "/data/" + g_name + "/val_node_set", data=val_node_set
                )
            if self.test_generator is not None:
                test_node_set = self.test_generator.sampler.avail_node_set[g_name]
                h5file.create_dataset(
                    "/data/" + g_name + "/test_node_set", data=test_node_set
                )

    def load_hdf5_data(self, h5file):
        train_node_set = dict()
        val_node_set = dict()
        test_node_set = dict()
        for k, v in h5file["/data/"].items():
            self.generator.graphs[k] = v["adj_matrix"][...]
            self.generator.inputs[k] = v["inputs"][...]
            self.generator.targets[k] = v["targets"][...]
            train_node_set[k] = v["train_node_set"][...]
            if "val_node_set" in v:
                val_node_set[k] = v["val_node_set"][...]

            if "test_node_set" in v:
                test_node_set[k] = v["test_node_set"][...]

        self.generator.sampler.update(self.generator.graphs, self.generator.inputs)

        if len(val_node_set) > 0:
            self.val_generator = copy.deepcopy(self.generator)
            self.val_generator.sampler = dl.generators.SequentialSampler()
            self.val_generator.sampler.avail_node_set = val_node_set

        if len(test_node_set) > 0:
            self.test_generator = copy.deepcopy(self.generator)
            self.test_generator.sampler = dl.generators.SequentialSampler()
            self.test_generator.sampler.avail_node_set = test_node_set

    def save_hdf5_all(self, h5file):
        h5file.create_dataset("/np_seed", data=self.np_seed)
        h5file.create_dataset("/tf_seed", data=self.tf_seed)
        h5file.create_dataset("/name/", data=self.name)
        self.save_hdf5_model(h5file)
        self.save_hdf5_optimizer(h5file)
        self.save_hdf5_history(h5file)
        self.save_hdf5_data(h5file)
