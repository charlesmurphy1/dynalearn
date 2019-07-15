import dynalearn as dl
import numpy as np
import networkx as nx
import time
import tqdm
import copy


class DynamicsGenerator:
    def __init__(
        self,
        graph_model,
        dynamics_model,
        sampler,
        batch_size=-1,
        with_truth=False,
        verbose=1,
    ):
        self.graph_model = graph_model
        self.dynamics_model = dynamics_model
        self.num_nodes = graph_model.num_nodes
        self.num_states = dynamics_model.num_states
        if sampler is None:
            self._sampler = dl.generators.RandomSampler()
        else:
            self._sampler = sampler
        self.batch_size = batch_size
        self.graphs = dict()
        self.inputs = dict()
        self.targets = dict()
        self.gt_targets = dict()

        self.with_truth = with_truth
        self.verbose = verbose

    def __len__(self):
        return np.sum([self.sampler.num_samples[n] for n in self.graphs])

    def __iter__(self):
        return self

    def __next__(self):
        g_index, s_index, n_mask = self.sampler(self.batch_size)
        inputs = self.inputs[g_index][s_index, :]
        adj = self.graphs[g_index]
        if self.with_truth:
            targets = self.gt_targets[g_index][s_index, :, :]
        else:
            targets = self.to_one_hot(self.targets[g_index][s_index, :])
        weights = n_mask
        return [inputs, adj], targets, weights

    def generate(self, num_sample, T, max_null_iter=100, shuffle=True):

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
            for t in range(T):
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

                if sample == num_sample or null_iteration == max_null_iter:
                    break

        if self.verbose:
            p_bar.close()

        if shuffle:
            index = np.random.permutation(num_sample)
        else:
            index = np.arange(num_sample)

        self.graphs[name] = adj
        self.inputs[name] = inputs[index, :]
        self.targets[name] = targets[index, :]
        self.gt_targets[name] = gt_targets[index, :]
        self.sampler.update(self.graphs, self.inputs)

    def _update_states(self):
        inputs = self.dynamics_model.states
        self.dynamics_model.update()
        targets = self.dynamics_model.states
        gt_targets = self.dynamics_model.predict(inputs)
        return inputs, targets, gt_targets

    def to_one_hot(self, arr):
        ans = np.zeros((arr.shape[0], self.num_states), dtype="int")
        ans[np.arange(arr.shape[0]), arr.astype("int")] = 1
        return ans

    def parition_generator(self, fraction, bias=1):
        gen_partition = copy.deepcopy(self)
        for i in self.graphs:
            num_nodes = self.graphs[i].shape[0]
            size = int(np.ceil(fraction * num_nodes))
            nodesubset = dict()
            for j in range(self.inputs[i].shape[0]):
                nodesubset = self.sampler.sample_nodes(i, j, size, bias).astype("int")
                gen_partition.sampler.avail_node_set[i][j] = nodesubset
                self.sampler.avail_node_set[i][j] = np.setdiff1d(
                    self.sampler.avail_node_set[i][j], nodesubset
                )

        self.sampler.update_weights(self.graphs, self.inputs)
        gen_partition.sampler.update_weights(self.graphs, self.inputs)
        gen_partition.batch_size = -1
        return gen_partition

    @property
    def sampler(self):
        return self._sampler

    @sampler.setter
    def sampler(self, sampler):
        avail_node_set = self._sampler.avail_node_set
        self._sampler = sampler
        self._sampler.update(self.graphs, self.inputs)
        self._sampler.avail_node_set = avail_node_set
