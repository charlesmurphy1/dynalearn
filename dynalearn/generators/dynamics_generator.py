import numpy as np
import networkx as nx
import time
import tqdm
import copy


class DynamicsGenerator:
    def __init__(
        # self, graph_model, dynamics_model, sampler, with_truth=False, verbose=1
        self,
        graph_model,
        dynamics_model,
        with_truth=False,
        verbose=1,
    ):
        self.graph_model = graph_model
        self.dynamics_model = dynamics_model
        self.num_states = dynamics_model.num_states
        # if sampler is None:
        #     self.sampler = SequentialSampler(self.num_states, 1)
        # else:
        #     self.sampler = sampler
        self.sample_size = 0
        self.graphs = dict()
        self.inputs = dict()
        self.targets = dict()

        self.with_truth = with_truth
        self.verbose = verbose

    def __len__(self):
        return self.sample_size

    def __iter__(self):
        return self

    def __next__(self):
        # g_index, s_index, n_mask = self.sampler()

        g_index = np.random.choice(np.array(list(self.graphs.keys())))
        s_index = np.random.randint(self.inputs[g_index].shape[0])
        inputs = self.inputs[g_index][s_index]
        adj = self.graphs[g_index]
        targets = self.targets[g_index][s_index]
        # weights = n_mask
        # return [inputs, adj], targets, weights
        return [inputs, adj], targets

    def generate(self, num_sample, T, max_null_iter=100):

        sample = 0
        name, graph = self.graph_model.generate()
        N = graph.number_of_nodes()

        adj = nx.to_numpy_array(graph)
        inputs = np.zeros([num_sample, N])
        targets = np.zeros([num_sample, N, self.num_states])
        # if self.with_truth:
        #     targets = np.zeros([num_sample, N, self.num_states])
        # else:
        #     targets = np.zeros([num_sample, N])

        self.dynamics_model.graph = graph

        if self.verbose:
            p_bar = tqdm.tqdm(range(num_sample), "Generating data")

        while sample < num_sample:
            self.dynamics_model.initialize_states()
            null_iteration = 0
            for t in range(T):
                t0 = time.time()
                x, y = self._update_states()

                inputs[sample] = x
                targets[sample] = y

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

        self.sample_size += num_sample
        self.graphs[name] = adj
        self.inputs[name] = inputs
        self.targets[name] = targets

        # self.sampler.update(self.graphs, self.inputs, self.targets)

    def _update_states(self):
        inputs = self.dynamics_model.states
        self.dynamics_model.update()
        if self.with_truth:
            targets = self.dynamics_model.predict(inputs)
        else:
            targets = self.to_one_hot(self.dynamics_model.states)
        return inputs, targets

    def to_one_hot(self, arr):
        ans = np.zeros((arr.shape[0], self.num_states), dtype="int")
        ans[np.arange(arr.shape[0]), arr.astype("int")] = 1
        return ans

    # def parition_generator(self, fraction):
    #     gen_parition = copy.deepcopy(self)
    #     for i in self.graphs:
    #         num_nodes = self.graphs[i].shape[0]
    #         n = int(np.ceil(fraction * num_nodes))
    #         nodesubset = self.sampler.sample_nodes(i, n).astype("int")
    #         gen_parition.sampler.avail_nodeset[i] = nodesubset
    #         self.sampler.avail_nodeset[i] = np.setdiff1d(
    #             self.sampler.avail_nodeset[i], nodesubset
    #         )
    #     gen_parition.sampler.params["batchsize"] = None
    #
    #     return gen_parition
