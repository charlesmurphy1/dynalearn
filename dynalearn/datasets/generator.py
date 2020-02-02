from .samplers import RandomSampler
import numpy as np
import networkx as nx
import time
import tqdm
from tensorflow.python.keras.utils.data_utils import Sequence


class DynamicsGenerator(Sequence):
    def __init__(self, graph_model, dynamics_model, sampler, config, verbose=0):
        self.__config = config
        self.graph_model = graph_model
        self.dynamics_model = dynamics_model
        self.num_states = dynamics_model.num_states
        self._samplers = {"train": sampler}
        self.mode = "train"

        self.batch_size = config.batch_size
        self.resampling_time = config.resampling_time
        self.max_null_iter = config.max_null_iter
        self.shuffle = config.shuffle
        self.with_truth = config.with_truth

        self.clear(clear_samplers=False)

        self.verbose = verbose

    def __len__(self):
        return np.sum([self.main_sampler.num_samples[n] for n in self.graphs])

    def __iter__(self):
        return self

    def __next__(self):
        g_index, s_index, n_mask = self.main_sampler(self.batch_size)
        inputs = self.inputs[g_index][s_index, :]
        adj = self.graphs[g_index]
        if self.with_truth:
            targets = self.gt_targets[g_index][s_index, :, :]
        else:
            targets = self.to_one_hot(self.targets[g_index][s_index, :])
        weights = n_mask
        # weights /= weights.sum()
        # weights *= weights.shape[0]
        return [inputs, adj], targets, weights

    def __getitem__(self, index):
        return self.__next__()

    def copy(self):

        generator_copy = self.__class__(
            self.graph_model, self.dynamics_model, None, self.__config, self.verbose
        )

        for k, s in self._samplers.items():
            generator_copy._samplers[k] = s.copy()

        generator_copy.graphs = self.graphs
        generator_copy.inputs = self.inputs
        generator_copy.targets = self.targets
        generator_copy.gt_targets = self.gt_targets
        return generator_copy

    def clear(self, clear_samplers=True):
        self.graphs = dict()
        self.inputs = dict()
        self.targets = dict()
        self.gt_targets = dict()

        if clear_samplers:
            for s in self.samplers:
                self.samplers[s].clear()

    def generate(self, num_sample):

        sample = 0
        name, graph = self.graph_model.generate()
        N = graph.number_of_nodes()

        adj = nx.to_numpy_array(graph)
        inputs = np.zeros([num_sample, N])
        targets = np.zeros([num_sample, N])
        gt_targets = np.zeros([num_sample, N, self.num_states])

        states = self.dynamics_model.initial_states(graph)

        if self.verbose == 1:
            p_bar = tqdm.tqdm(range(num_sample))

        while sample < num_sample:
            null_iteration = 0
            for t in range(self.resampling_time):
                t0 = time.time()
                x, y, z = self.__update_states(states)

                inputs[sample, :] = x
                targets[sample, :] = y
                gt_targets[sample] = z

                states = y

                t1 = time.time()

                if self.verbose == 1:
                    p_bar.set_description(
                        "Generating data - " + str(round(t1 - t0, 5)) + "s"
                    )
                    p_bar.update()

                sample += 1
                if self.dynamics_model.is_dead(states):
                    null_iteration += 1

                if sample == num_sample or null_iteration == self.max_null_iter:
                    break
            self.dynamics_model.initial_states()

        if self.verbose == 1:
            p_bar.close()

        if self.shuffle:
            index = np.random.permutation(num_sample)
        else:
            index = np.arange(num_sample)

        self.graphs[name] = adj
        self.inputs[name] = inputs[index, :]
        self.targets[name] = targets[index, :]
        self.gt_targets[name] = gt_targets[index, :]
        self.main_sampler.update(self.graphs, self.inputs)

    def __update_states(self, states):
        inputs = states
        targets = self.dynamics_model.sample(states)
        gt_targets = self.dynamics_model.predict(inputs)
        return inputs, targets, gt_targets

    def to_one_hot(self, arr):
        ans = np.zeros((arr.shape[0], self.num_states), dtype="int")
        ans[np.arange(arr.shape[0]), arr.astype("int")] = 1
        return ans

    def partition_sampler(self, name, fraction=None, bias=1):
        new_sampler = self.main_sampler.copy()
        new_sampler.name = name
        if fraction is not None:
            for g in self.graphs:
                num_nodes = self.graphs[g].shape[0]
                size = int(np.ceil(fraction * num_nodes))
                for t in range(self.inputs[g].shape[0]):
                    nodesubset = self.main_sampler.sample_nodes(
                        g, t, size, bias
                    ).astype("int")
                    new_sampler.avail_node_set[g][t] = nodesubset
                    self.main_sampler.avail_node_set[g][t] = np.setdiff1d(
                        self.main_sampler.avail_node_set[g][t], nodesubset
                    )

            self.main_sampler.update_weights(self.graphs, self.inputs)
            new_sampler.update_weights(self.graphs, self.inputs)
        self.samplers[name] = new_sampler

    # @property
    # def shape(self):
    #     return (len(self),)

    @property
    def samplers(self):
        return self._samplers

    @samplers.setter
    def samplers(self, samplers):
        for k, v in samplers.items():
            avail_node_set = self._samplers[k].avail_node_set
            self._samplers[k] = v
            self._samplers[k].update(self.graphs, self.inputs)
            self._samplers[k].avail_node_set = avail_node_set

    @property
    def main_sampler(self):
        return self._samplers[self.mode]

    def save(self, h5file, overwrite=False):
        graph_name = type(self.graph_model).__name__
        graph_params = self.graph_model.params

        dynamics_name = type(self.dynamics_model).__name__
        dynamics_params = self.dynamics_model.params

        if "data" in h5file:
            if overwrite:
                del h5file["data"]
            else:
                return
        h5file.create_group("data")
        h5group = h5file["data"]

        for g_name in self.graphs:
            adj = self.graphs[g_name]
            inputs = self.inputs[g_name]
            targets = self.targets[g_name]
            gt_targets = self.gt_targets[g_name]
            if g_name in h5file:
                if overwrite:
                    del h5file[g_name]
                else:
                    continue
            h5group.create_dataset(g_name + "/adj_matrix", data=adj)
            h5group.create_dataset(g_name + "/inputs", data=inputs)
            h5group.create_dataset(g_name + "/targets", data=targets)
            h5group.create_dataset(g_name + "/gt_targets", data=gt_targets)

        for k in self.samplers:
            self.samplers[k].save(h5file, overwrite)

    def load(self, h5file):
        if "data" in h5file:
            for k, v in h5file["data"].items():
                self.graphs[k] = v["adj_matrix"][...]
                self.inputs[k] = v["inputs"][...]
                self.targets[k] = v["targets"][...]
                self.gt_targets[k] = v["gt_targets"][...]

        if "sampler" in h5file:
            for k in h5file["sampler"]:
                if k not in self.samplers:
                    self.partition_sampler(k)
                self.samplers[k].load(h5file)
