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
        self.np_seed = numpy_seed
        self.tf_seed = tensorflow_seed
        self.verbose = verbose

    def generate_data(
        self,
        num_graphs,
        num_sample,
        T,
        val_fraction=None,
        test_fraction=None,
        val_bias=0,
        test_bias=0,
        **kwargs
    ):
        for i in range(num_graphs):
            self.generator.generate(num_sample, T, **kwargs)

        if val_fraction is not None:
            if self.verbose:
                print("Partitioning generator for validation")
            self.val_generator = self.generator.parition_generator(
                val_fraction, val_bias
            )

        if test_fraction is not None:
            if self.verbose:
                print("Partitioning generator for test")
            self.test_generator = self.generator.parition_generator(
                test_fraction, test_bias
            )

    def train_model(
        self,
        epochs,
        steps_per_epoch,
        validation_steps=0,
        metrics=[],
        callbacks=[],
        learning_rate=1e-3,
    ):
        if self.val_generator is None:
            validation_steps = None
        i_epoch = self.epoch
        f_epoch = self.epoch + epochs

        self.optimizer.lr = K.variable(learning_rate)
        self.model.model.compile(self.optimizer, self.loss, metrics)
        history = self.model.model.fit_generator(
            self.generator,
            validation_data=self.val_generator,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            initial_epoch=i_epoch,
            epochs=f_epoch,
            verbose=self.verbose,
            callbacks=callbacks,
        )

        for k in history.history:
            if k not in self.history:
                self.history[k] = [np.nan] * i_epoch
            elif len(self.history[k]) < i_epoch:
                to_fill = i_epoch - len(self.history[k])
                self.history[k].extend([np.nan] * to_fill)
            self.history[k].extend(history.history[k])

        self.epoch += epochs

        return history

    def save_weights(self, filepath, overwrite=True):
        self.model.model.save_weights(filepath)

    def load_weights(self, filepath):
        self.model.model.load_weights(filepath)
        return 0

    def save_history(self, filepath, overwrite=True):
        h5file = h5py.File(filepath)
        if "history" in h5file:
            if not overwrite:
                return
            else:
                del h5file["history"]
        for name, value in self.history.items():
            h5file.create_dataset("history/" + name, data=value, fillvalue=np.nan)
        h5file.close()

    def load_history(self, filepath):
        h5file = h5py.File(filepath)
        if "history" in h5file:
            for k, v in h5file["history"].items():
                self.history[k] = list(v[...])
        h5file.close()

    def save_data(self, filepath, overwrite=True):
        graph_name = type(self.graph_model).__name__
        graph_params = self.graph_model.params

        dynamics_name = type(self.dynamics_model).__name__
        dynamics_params = self.dynamics_model.params

        h5file = h5py.File(filepath)
        if "data" in h5file:
            if overwrite:
                del h5file["data"]
            else:
                h5file.close()
                return

        for g_name in self.generator.graphs:
            adj = self.generator.graphs[g_name]
            inputs = self.generator.inputs[g_name]
            targets = self.generator.targets[g_name]
            if g_name in h5file:
                if overwrite:
                    del h5file[g_name]
                else:
                    continue
            h5file.create_dataset(g_name + "/adj_matrix", data=adj)
            h5file.create_dataset(g_name + "/inputs", data=inputs)
            h5file.create_dataset(g_name + "/targets", data=targets)

            # Training set
            avail_node_set = np.array(
                [
                    np.where(
                        self.generator.sampler.node_weights[g_name][i] > 0,
                        np.ones(self.generator.sampler.node_weights[g_name][i].shape),
                        np.zeros(self.generator.sampler.node_weights[g_name][i].shape),
                    )
                    for i in range(inputs.shape[0])
                ]
            )
            node_weights = np.array(
                [
                    self.generator.sampler.node_weights[g_name][i]
                    for i in range(inputs.shape[0])
                ]
            )
            state_weights = np.array(
                [
                    self.generator.sampler.state_weights[g_name][i]
                    for i in range(inputs.shape[0])
                ]
            )
            graph_weights = self.generator.sampler.graph_weights[g_name]
            h5file.create_dataset(
                g_name + "/training_set/avail_node_set", data=avail_node_set
            )
            h5file.create_dataset(
                g_name + "/training_set/node_weights", data=node_weights
            )
            h5file.create_dataset(
                g_name + "/training_set/state_weights", data=state_weights
            )
            h5file.create_dataset(
                g_name + "/training_set/graph_weights", data=graph_weights
            )

            if self.val_generator is not None:
                # Validation set
                avail_node_set = np.array(
                    [
                        np.where(
                            self.val_generator.sampler.node_weights[g_name][i] > 0,
                            np.ones(
                                self.val_generator.sampler.node_weights[g_name][i].shape
                            ),
                            np.zeros(
                                self.val_generator.sampler.node_weights[g_name][i].shape
                            ),
                        )
                        for i in range(inputs.shape[0])
                    ]
                )
                node_weights = np.array(
                    [
                        self.val_generator.sampler.node_weights[g_name][i]
                        for i in range(inputs.shape[0])
                    ]
                )
                state_weights = np.array(
                    [
                        self.val_generator.sampler.state_weights[g_name][i]
                        for i in range(inputs.shape[0])
                    ]
                )
                graph_weights = self.val_generator.sampler.graph_weights[g_name]
                h5file.create_dataset(
                    g_name + "/validation_set/avail_node_set", data=avail_node_set
                )
                h5file.create_dataset(
                    g_name + "/validation_set/node_weights", data=node_weights
                )
                h5file.create_dataset(
                    g_name + "/validation_set/state_weights", data=state_weights
                )
                h5file.create_dataset(
                    g_name + "/validation_set/graph_weights", data=graph_weights
                )
            if self.test_generator is not None:
                # Training set
                avail_node_set = np.array(
                    [
                        np.where(
                            self.test_generator.sampler.node_weights[g_name][i] > 0,
                            np.ones(
                                self.test_generator.sampler.node_weights[g_name][
                                    i
                                ].shape
                            ),
                            np.zeros(
                                self.test_generator.sampler.node_weights[g_name][
                                    i
                                ].shape
                            ),
                        )
                        for i in range(inputs.shape[0])
                    ]
                )
                node_weights = np.array(
                    [
                        self.test_generator.sampler.node_weights[g_name][i]
                        for i in range(inputs.shape[0])
                    ]
                )
                state_weights = np.array(
                    [
                        self.test_generator.sampler.state_weights[g_name][i]
                        for i in range(inputs.shape[0])
                    ]
                )
                graph_weights = self.test_generator.sampler.graph_weights[g_name]
                h5file.create_dataset(
                    g_name + "/test_set/avail_node_set", data=avail_node_set
                )
                h5file.create_dataset(
                    g_name + "/test_set/node_weights", data=node_weights
                )
                h5file.create_dataset(
                    g_name + "/test_set/state_weights", data=state_weights
                )
                h5file.create_dataset(
                    g_name + "/test_set/graph_weights", data=graph_weights
                )

        h5file.close()

    def load_data(self, path):

        h5file = h5py.File(path)
        if "data" in h5file:
            for k, v in h5file["data"].items():
                self.generator.graphs[k] = v["adj_matrix"][...]
                self.generator.inputs[k] = v["inputs"][...]
                self.generator.targets[k] = v["targets"][...]

                avail_node_set = v["training_set/avail_node_set"][...]
                node_weights = v["training_set/node_weights"][...]
                state_weights = v["training_set/state_weights"][...]
                graph_weights = v["training_set/graph_weights"][...]
                self.generator.sampler.avail_node_set[k] = {
                    i: np.argwhere(nodes) for i, nodes in enumerate(avail_node_set)
                }
                self.generator.sampler.node_weights[k] = {
                    i: weights for i, weights in enumerate(node_weights)
                }
                self.generator.sampler.state_weights[k] = {
                    i: weights for i, weights in enumerate(state_weights)
                }
                self.generator.sampler.graph_weights[k] = graph_weights

                if "validation_set" in v:
                    self.val_generator = copy.deepcopy(self.generator)
                    avail_node_set = v["validation_set/avail_node_set"][...]
                    node_weights = v["validation_set/node_weights"][...]
                    state_weights = v["validation_set/state_weights"][...]
                    graph_weights = v["validation_set/graph_weights"][...]
                    self.val_generator.sampler.avail_node_set[k] = {
                        i: np.argwhere(nodes) for i, nodes in enumerate(avail_node_set)
                    }
                    self.val_generator.sampler.node_weights[k] = {
                        i: weights for i, weights in enumerate(node_weights)
                    }
                    self.val_generator.sampler.state_weights[k] = {
                        i: weights for i, weights in enumerate(state_weights)
                    }
                    self.val_generator.sampler.graph_weights[k] = graph_weights

                if "test_set" in v:
                    self.test_generator = copy.deepcopy(self.generator)
                    avail_node_set = v["test_set/avail_node_set"][...]
                    node_weights = v["test_set/node_weights"][...]
                    state_weights = v["test_set/state_weights"][...]
                    graph_weights = v["test_set/graph_weights"][...]
                    self.test_generator.sampler.avail_node_set[k] = {
                        i: np.argwhere(nodes) for i, nodes in enumerate(avail_node_set)
                    }
                    self.test_generator.sampler.node_weights[k] = {
                        i: weights for i, weights in enumerate(node_weights)
                    }
                    self.test_generator.sampler.state_weights[k] = {
                        i: weights for i, weights in enumerate(state_weights)
                    }
                    self.test_generator.sampler.graph_weights[k] = graph_weights
        h5file.close()
