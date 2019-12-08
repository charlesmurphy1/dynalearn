import dynalearn as dl
import numpy as np
import networkx as nx
import time
import tqdm
import copy


class DynamicsGenerator:
    def __init__(self, graph_model, dynamics_model, sampler, config, verbose=0):
        self.graph_model = graph_model
        self.dynamics_model = dynamics_model
        self.num_states = dynamics_model.num_states
        if sampler is None:
            self._samplers = {"train": dl.generators.RandomSampler("train")}
        else:
            sampler.name = "train"
            self._samplers = {"train": sampler}
        self.mode = "train"

        self.batch_size = config.batch_size
        self.resampling_time = config.resampling_time
        self.max_null_iter = config.max_null_iter
        self.shuffle = config.shuffle
        self.with_truth = config.with_truth

        self.graphs = dict()
        self.inputs = dict()
        self.targets = dict()
        self.gt_targets = dict()

        self.verbose = verbose

    def __len__(self):
        return np.sum([self.samplers[self.mode].num_samples[n] for n in self.graphs])

    def __iter__(self):
        return self

    def __next__(self):
        g_index, s_index, n_mask = self.samplers[self.mode](self.batch_size)
        inputs = self.inputs[g_index][s_index, :]
        adj = self.graphs[g_index]
        if self.with_truth:
            targets = self.gt_targets[g_index][s_index, :, :]
        else:
            targets = self.to_one_hot(self.targets[g_index][s_index, :])
        weights = n_mask
        return [inputs, adj], targets, weights

    def generate(self, num_sample):

        sample = 0
        name, graph = self.graph_model.generate()
        N = graph.number_of_nodes()

        adj = nx.to_numpy_array(graph)
        inputs = np.zeros([num_sample, N])
        targets = np.zeros([num_sample, N])
        gt_targets = np.zeros([num_sample, N, self.num_states])

        self.dynamics_model.graph = graph

        if self.verbose:
            p_bar = tqdm.tqdm(range(num_sample), "Generating data")

        while sample < num_sample:
            self.dynamics_model.initialize_states()
            null_iteration = 0
            for t in range(self.resampling_time):
                t0 = time.time()
                x, y, z = self._update_states()

                inputs[sample, :] = x
                targets[sample, :] = y
                gt_targets[sample] = z

                t1 = time.time()

                if self.verbose:
                    p_bar.set_description(
                        "Generating data - " + str(round(t1 - t0, 5)) + "s"
                    )
                    p_bar.update()

                sample += 1
                if not self.dynamics_model.continue_simu:
                    null_iteration += 1

                if sample == num_sample or null_iteration == self.max_null_iter:
                    break

        if self.verbose:
            p_bar.close()

        if self.shuffle:
            index = np.random.permutation(num_sample)
        else:
            index = np.arange(num_sample)

        self.graphs[name] = adj
        self.inputs[name] = inputs[index, :]
        self.targets[name] = targets[index, :]
        self.gt_targets[name] = gt_targets[index, :]
        self.samplers[self.mode].update(self.graphs, self.inputs)

    def _update_states(self):
        inputs = self.dynamics_model.states
        targets = self.dynamics_model.sample()
        gt_targets = self.dynamics_model.predict(inputs)
        return inputs, targets, gt_targets

    def to_one_hot(self, arr):
        ans = np.zeros((arr.shape[0], self.num_states), dtype="int")
        ans[np.arange(arr.shape[0]), arr.astype("int")] = 1
        return ans

    def partition_sampler(self, name, fraction=None, bias=1):
        self.samplers[name] = copy.deepcopy(self.samplers[self.mode])
        self.samplers[name].name = name

        if fraction is not None:
            for i in self.graphs:
                num_nodes = self.graphs[i].shape[0]
                size = int(np.ceil(fraction * num_nodes))
                for j in range(self.inputs[i].shape[0]):
                    nodesubset = (
                        self.samplers[self.mode]
                        .sample_nodes(i, j, size, bias)
                        .astype("int")
                    )
                    self.samplers[name].avail_node_set[i][j] = nodesubset
                    self.samplers[self.mode].avail_node_set[i][j] = np.setdiff1d(
                        self.samplers[self.mode].avail_node_set[i][j], nodesubset
                    )

            self.samplers[self.mode].update_weights(self.graphs, self.inputs)
            self.samplers[name].update_weights(self.graphs, self.inputs)

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
